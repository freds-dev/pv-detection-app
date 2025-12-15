import flask
from flask import request, jsonify, render_template
import yaml
import os
import logging
import uuid
import json
from joblib import load
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Point
from sqlalchemy import create_engine, text
import rasterio
import shutil

# Import our refactored utility functions
import prediction_utils as pu

# --- App Setup ---
app = flask.Flask(__name__)
pu.setup_logging()
LOGGER = logging.getLogger("pv-flask-app")

# --- Load Config ---
try:
    with open("config.yaml", "r") as f:
        CFG = yaml.safe_load(f)
        LOGGER.info("Config.yaml loaded successfully.")
except FileNotFoundError:
    LOGGER.error("CRITICAL: config.yaml not found. Please create one.")
    exit(1)
except Exception as e:
    LOGGER.error(f"CRITICAL: Error loading config.yaml: {e}")
    exit(1)

# --- Database Connection ---
try:
    db_cfg = CFG['database']
    DB_URL = f"postgresql+psycopg2://{db_cfg['user']}:{db_cfg['password']}@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}"
    ENGINE = create_engine(DB_URL)
    with ENGINE.connect() as conn:
        conn.execute(text("SELECT 1"))
    LOGGER.info(f"Successfully connected to database: {db_cfg['dbname']}")
except Exception as e:
    LOGGER.error(f"CRITICAL: Database connection failed: {e}")
    LOGGER.error(
        "Please check your config.yaml and ensure PostGIS is running.")
    exit(1)

# --- Load Model ---
try:
    MODEL_PATH = CFG['prediction']['model_path']
    MODEL = load(MODEL_PATH)
    THRESHOLD = 0.5  # You might want to load this from metrics.json
    LOGGER.info(f"Successfully loaded model from: {MODEL_PATH}")
except Exception as e:
    LOGGER.error(f"CRITICAL: Failed to load model from {MODEL_PATH}: {e}")
    exit(1)

# === Frontend Route ===


@app.route('/')
def index():
    """Serves the main HTML map page."""
    # Pass config values to the frontend
    frontend_config = {
        'planet_api_key': CFG['api_keys'].get('planet_labs')
    }
    return render_template('index.html', config=frontend_config)

# === API Endpoints ===


@app.route('/api/find_building', methods=['POST'])
def find_building():
    """
    Finds the closest building to a clicked point.
    """
    try:
        data = request.json
        lon, lat = data['lon'], data['lat']

        # CRS 4326 is standard for lat/lon
        clicked_point_wkt = f"POINT({lon} {lat})"
        tbl = db_cfg['footprints_table']
        geom_col = db_cfg['geometry_column']

        # We run TWO fast queries instead of ONE slow one.

        with ENGINE.connect() as conn:

            # Step 1: Get the native SRID of the geometry column ONCE.
            # This is a very fast metadata query.
            srid_query = text(
                f"SELECT ST_SRID(t.{geom_col}) FROM {tbl} t LIMIT 1")
            srid_result = conn.execute(srid_query)
            native_srid = srid_result.fetchone()[0]

            if not native_srid:
                raise Exception(f"Could not determine SRID for table {tbl}")

            # Step 2: Run the main k-NN query, passing the SRID as a static parameter.
            # This allows the query planner to use the GIST index perfectly.
            query = text(f"""
                SELECT 
                    json_build_object('type', 'Feature', 'geometry', ST_AsGeoJSON(ST_Transform(t.{geom_col}, 4326))::json) as geojson
                FROM 
                    {tbl} t
                ORDER BY 
                    t.{geom_col} <-> ST_Transform(ST_GeomFromText(:point_wkt, 4326), :native_srid)
                LIMIT 1;
            """)

            # Pass *both* parameters to the query
            result = conn.execute(
                query,
                {"point_wkt": clicked_point_wkt, "native_srid": native_srid}
            )
            row = result.fetchone()

        if row:
            return jsonify({"status": "success", "building": row[0]})
        else:
            return jsonify({"status": "not_found", "message": "No building found near that point."}), 404

    except Exception as e:
        LOGGER.error(f"Error in /api/find_building: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Runs prediction. Optionally calculates uncertainty (DI/LPD) based on user flag.
    """
    try:
        data = request.json
        building_geojson = data['building_geojson']
        # Default to True if not provided, or strictly follow the frontend
        calculate_uncertainty = data.get('calculate_uncertainty', True)

        building_geom = shape(building_geojson['geometry'])

        # --- Setup ---
        session_id = str(uuid.uuid4())
        results_dir = os.path.join('static', 'results', session_id)
        os.makedirs(results_dir, exist_ok=True)

        r_script_dir = os.path.abspath(results_dir)
        r_script_path = os.path.abspath("scripts/calculate_di.R")

        pred_cfg = CFG['prediction']
        target_epsg = 3857

        # 1. Create AOI
        gdf_aoi, bbox_target, bbox_4326 = pu.get_aoi_from_building_geom(
            building_geom, target_epsg, pred_cfg['buffer_m']
        )

        # 2. Find Sentinel-2 STAC Item
        stac_item = pu.stac_search_one(pred_cfg, bbox_4326)

        # 3. Read Sentinel Bands
        stac_bands = sorted(set(getattr(MODEL, "feature_names_in_", [])) | {
                            "B04", "B03", "B02"})
        stac_bands = [b for b in stac_bands if b not in ['x', 'y']]

        band_arrays, transform, dst_crs = pu.read_bands_to_grid(
            stac_item, stac_bands, bbox_target, target_epsg, resolution=10.0
        )
        H, W = next(iter(band_arrays.values())).shape

        # 4. Create Building Mask
        building_gdf = gpd.GeoDataFrame(
            [{'geometry': building_geom}], crs=4326)
        building_mask = pu.create_building_mask(
            building_gdf, transform, (H, W), dst_crs)

        # 5. Assemble Features & Predict
        X, feature_names = pu.assemble_features(MODEL, band_arrays)
        proba, pred = pu.predict_mask(MODEL, pd.DataFrame(
            X, columns=feature_names), (H, W), THRESHOLD)

        # 6. Apply Mask to Visuals
        proba[building_mask == 0] = np.nan
        pred[building_mask == 0] = 0

        # --- 90th Percentile Classification ---
        valid_pixels = proba[building_mask == 1]
        valid_pixels = valid_pixels[~np.isnan(valid_pixels)]

        if len(valid_pixels) == 0:
            pv_score = 0.0
        else:
            pv_score = float(np.percentile(valid_pixels, 90))

        is_pv = pv_score >= 0.5
        cls_text = "PV Detected" if is_pv else "No PV Detected"
        # --------------------------------------

        # 7. Save Standard Visual Layers
        layer_urls = {}
        png_bounds = tuple(gpd.GeoSeries(
            gdf_aoi, crs=target_epsg).to_crs(4326).total_bounds)
        png_bounds_leaflet = [[png_bounds[1], png_bounds[0]], [
            png_bounds[3], png_bounds[2]]]

        try:
            heat_png_name = "pv_probability.png"
            heat_png_path = os.path.join(results_dir, heat_png_name)
            pu.save_png_heatmap(proba, heat_png_path)
            layer_urls['pv_probability'] = f"/{results_dir}/{heat_png_name}"
        except Exception as e:
            LOGGER.warning(f"Failed to create heatmap: {e}")

        try:
            rgb_png_name = "sentinel_true_color.png"
            rgb_png_path = os.path.join(results_dir, rgb_png_name)
            pu.save_png_rgb(band_arrays, rgb_png_path)
            layer_urls['sentinel_true_color'] = f"/{results_dir}/{rgb_png_name}"
        except Exception as e:
            LOGGER.warning(f"Failed to create RGB: {e}")

        # --- 8. CONDITIONAL Uncertainty Estimation (DI/LPD) ---
        if calculate_uncertainty:
            try:
                # Prepare data for R script
                prediction_features_path = os.path.join(
                    r_script_dir, "prediction_features.csv")
                pd.DataFrame(X, columns=feature_names).to_csv(
                    prediction_features_path, index=False)

                mask_path = os.path.join(r_script_dir, "building_mask.tif")
                with rasterio.open(
                    mask_path, 'w', driver='GTiff', height=H, width=W, count=1,
                    dtype=building_mask.dtype, crs=dst_crs, transform=transform
                ) as dst:
                    dst.write(building_mask, 1)

                meta_path = os.path.join(r_script_dir, "spatial_meta.json")
                with open(meta_path, "w") as f:
                    json.dump({"height": H, "width": W, "crs_wkt": dst_crs.to_wkt(),
                               "transform": transform.to_gdal()}, f)

                try:
                    source_data_path = CFG['prediction']['original_training_data_csv']
                    dest_data_path = os.path.join(
                        r_script_dir, "training_data.csv")
                    if not os.path.exists(source_data_path):
                        LOGGER.warning(
                            f"Training data missing: {source_data_path}. Skipping DI.")
                        di_lpd_available = False
                    else:
                        shutil.copyfile(source_data_path, dest_data_path)
                        # Run R script
                        di_lpd_available = pu.run_r_script(
                            r_script_path, r_script_dir)
                except Exception as e:
                    LOGGER.error(f"Error prepping DI data: {e}")
                    di_lpd_available = False

                if di_lpd_available:
                    di_tif_path = os.path.join(r_script_dir, "aoi_di.tif")
                    di_png_name = "dissimilarity_index.png"
                    di_png_path = os.path.join(results_dir, di_png_name)
                    pu.save_geotiff_as_colored_png(
                        di_tif_path, di_png_path, "RdYlGn_r", "DI")
                    layer_urls['dissimilarity_index'] = f"/{results_dir}/{di_png_name}"

                    lpd_tif_path = os.path.join(r_script_dir, "aoi_lpd.tif")
                    lpd_png_name = "local_point_density.png"
                    lpd_png_path = os.path.join(results_dir, lpd_png_name)
                    pu.save_geotiff_as_colored_png(
                        lpd_tif_path, lpd_png_path, "RdYlGn", "LPD")
                    layer_urls['local_point_density'] = f"/{results_dir}/{lpd_png_name}"

            except Exception as e:
                LOGGER.error(f"Failed to run R script: {e}", exc_info=True)
        else:
            LOGGER.info("Skipping Uncertainty Estimation (User Request)")

        # 9. Return JSON
        return jsonify({
            "status": "success",
            "layers": layer_urls,
            "bounds": png_bounds_leaflet,
            "classification": {
                "is_pv": is_pv,
                "score": round(pv_score, 3),
                "text": cls_text
            }
        })

    except Exception as e:
        LOGGER.error(f"Error in /api/predict: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host=CFG['server']['host'], port=CFG['server']['port'], debug=True)
