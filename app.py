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
# The MODEL variable is kept as None to maintain compatibility with utility functions.
THRESHOLD = 0.613
MODEL = None

# Check if the R model exists before starting the app to avoid runtime crashes
R_MODEL_PATH = os.path.abspath("data/artifacts/rf/model_final.rds")
if not os.path.exists(R_MODEL_PATH):
    LOGGER.warning(
        f"WARNING: R model not found at {R_MODEL_PATH}. Prediction will fail.")
else:
    LOGGER.info(f"App initialized to use R model at: {R_MODEL_PATH}")

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
    Runs prediction by calling an R script to use the .rds model.
    """
    try:
        data = request.json
        building_geojson = data['building_geojson']
        building_geom = shape(building_geojson['geometry'])

        # --- Setup Session ---
        session_id = str(uuid.uuid4())
        results_dir = os.path.join('static', 'results', session_id)
        os.makedirs(results_dir, exist_ok=True)
        r_script_dir = os.path.abspath(results_dir)

        r_script_path = os.path.abspath("scripts/predict.R")
        pred_cfg = CFG['prediction']
        target_epsg = 3857

        # 1. Create AOI (Area of Interest)
        gdf_aoi, bbox_target, bbox_4326 = pu.get_aoi_from_building_geom(
            building_geom, target_epsg, pred_cfg['buffer_m']
        )

        # 2. Find Sentinel-2 STAC Item
        stac_item = pu.stac_search_one(pred_cfg, bbox_4326)

        # 3. Read Sentinel Bands (Hardcoded to match your R model features)
        stac_bands = ["B02", "B03", "B04", "B05",
                      "B06", "B07", "B08", "B8A", "B11", "B12"]
        band_arrays, transform, dst_crs = pu.read_bands_to_grid(
            stac_item, stac_bands, bbox_target, target_epsg, resolution=10.0
        )
        H, W = next(iter(band_arrays.values())).shape

        # 4. Create Building Mask
        building_gdf = gpd.GeoDataFrame(
            [{'geometry': building_geom}], crs=4326)
        building_mask = pu.create_building_mask(
            building_gdf, transform, (H, W), dst_crs)

        # 5. Assemble Features & Save for R
        # We pass None for MODEL because we aren't using joblib anymore
        X, feature_names = pu.assemble_features(
            None, band_arrays, feature_list=stac_bands)
        pd.DataFrame(X, columns=feature_names).to_csv(
            os.path.join(r_script_dir, "prediction_features.csv"), index=False
        )

        # 6. Execute R Prediction
        success = pu.run_r_script(r_script_path, r_script_dir)
        if not success:
            raise Exception("R script failed to generate predictions.")

        # 7. Load R Results & Reshape
        preds_path = os.path.join(r_script_dir, "prediction_results.csv")
        r_preds = pd.read_csv(preds_path)

        # Determine the PV probability column (R labels are often 'PV' or 'X1')
        pv_col = [c for c in r_preds.columns if 'PV' in c or 'X1' in c][0]
        proba = r_preds[pv_col].values.reshape((H, W))

        # Apply mask (only show probability inside the building)
        proba[building_mask == 0] = np.nan

        # 8. Classification Logic
        valid_pixels = proba[building_mask == 1]
        valid_pixels = valid_pixels[~np.isnan(valid_pixels)]
        pv_score = float(np.percentile(valid_pixels, 90)
                         ) if len(valid_pixels) > 0 else 0.0
        is_pv = pv_score >= THRESHOLD

        # 9. Save Visual Layers
        layer_urls = {}
        png_bounds = tuple(gpd.GeoSeries(
            gdf_aoi, crs=target_epsg).to_crs(4326).total_bounds)
        png_bounds_leaflet = [[png_bounds[1], png_bounds[0]], [
            png_bounds[3], png_bounds[2]]]

        # Probability Heatmap
        pu.save_png_heatmap(proba, os.path.join(
            results_dir, "pv_probability.png"))
        layer_urls['pv_probability'] = f"/{results_dir}/pv_probability.png"

        # True Color Sentinel Image
        pu.save_png_rgb(band_arrays, os.path.join(
            results_dir, "sentinel_true_color.png"))
        layer_urls['sentinel_true_color'] = f"/{results_dir}/sentinel_true_color.png"

        return jsonify({
            "status": "success",
            "layers": layer_urls,
            "bounds": png_bounds_leaflet,
            "classification": {
                "is_pv": is_pv,
                "score": round(pv_score, 3),
                "text": "PV Detected" if is_pv else "No PV Detected"
            }
        })

    except Exception as e:
        LOGGER.error(f"Error in /api/predict: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host=CFG['server']['host'], port=CFG['server']['port'], debug=True)
