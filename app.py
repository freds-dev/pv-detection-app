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
import re

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
except Exception as e:
    LOGGER.error(f"CRITICAL: Error loading config.yaml: {e}")
    exit(1)

# --- Database Connection ---
try:
    db_cfg = CFG['database']
    DB_URL = f"postgresql+psycopg2://{db_cfg['user']}:{db_cfg['password']}@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}"
    ENGINE = create_engine(DB_URL)
    LOGGER.info(f"Successfully connected to database: {db_cfg['dbname']}")
except Exception as e:
    LOGGER.error(f"CRITICAL: Database connection failed: {e}")
    exit(1)

# --- Load Model ---
try:
    MODEL_PATH = CFG['prediction']['model_path']
    MODEL = load(MODEL_PATH)
    THRESHOLD = 0.5
    LOGGER.info(f"Successfully loaded model from: {MODEL_PATH}")
except Exception as e:
    LOGGER.error(f"CRITICAL: Failed to load model: {e}")
    exit(1)


@app.route('/')
def index():
    frontend_config = {'planet_api_key': CFG['api_keys'].get('planet_labs')}
    return render_template('index.html', config=frontend_config)


@app.route('/api/find_building', methods=['POST'])
def find_building():
    try:
        data = request.json
        lon, lat = data['lon'], data['lat']
        clicked_point_wkt = f"POINT({lon} {lat})"
        tbl, geom_col = db_cfg['footprints_table'], db_cfg['geometry_column']

        with ENGINE.connect() as conn:
            srid_query = text(
                f"SELECT ST_SRID(t.{geom_col}) FROM {tbl} t LIMIT 1")
            native_srid = conn.execute(srid_query).fetchone()[0]

            query = text(f"""
                SELECT json_build_object('type', 'Feature', 'geometry', ST_AsGeoJSON(ST_Transform(t.{geom_col}, 4326))::json)
                FROM {tbl} t
                ORDER BY t.{geom_col} <-> ST_Transform(ST_GeomFromText(:point_wkt, 4326), :native_srid)
                LIMIT 1;
            """)
            result = conn.execute(
                query, {"point_wkt": clicked_point_wkt, "native_srid": native_srid})
            row = result.fetchone()

        if row:
            return jsonify({"status": "success", "building": row[0]})
        return jsonify({"status": "not_found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        building_geojson = data['building_geojson']
        calculate_uncertainty = data.get('calculate_uncertainty', True)
        building_geom = shape(building_geojson['geometry'])

        session_id = str(uuid.uuid4())
        results_dir = os.path.join('static', 'results', session_id)
        os.makedirs(results_dir, exist_ok=True)

        r_script_path = os.path.abspath("scripts/calculate_di.R")
        pred_cfg = CFG['prediction']
        target_epsg = 3857

        # 1. AOI & STAC Search
        gdf_aoi, bbox_target, bbox_4326 = pu.get_aoi_from_building_geom(
            building_geom, target_epsg, pred_cfg['buffer_m'])
        stac_item = pu.stac_search_one(pred_cfg, bbox_4326)

        # 2. Identify required bands
        raw_features = list(getattr(MODEL, "feature_names_in_", []))
        stac_bands = sorted(set(pu.clean_str(f)
                            for f in raw_features) | {"B04", "B03", "B02"})
        stac_bands = [
            b for b in stac_bands if "season" not in b and b not in ['x', 'y']]

        # 3. Read Sentinel Bands
        band_arrays, transform, dst_crs = pu.read_bands_to_grid(
            stac_item, stac_bands, bbox_target, target_epsg)
        H, W = next(iter(band_arrays.values())).shape

        # 4. Assemble Features (Now handles seasons via stac_item.datetime)
        X, feature_names = pu.assemble_features(
            MODEL, band_arrays, stac_item.datetime)

        # 5. Predict
        proba, _ = pu.predict_mask(MODEL, X, (H, W), THRESHOLD)

        # 6. Apply Building Mask
        building_gdf = gpd.GeoDataFrame(
            [{'geometry': building_geom}], crs=4326)
        building_mask = pu.create_building_mask(
            building_gdf, transform, (H, W), dst_crs)
        proba[building_mask == 0] = np.nan

        # 7. Save Visuals
        layer_urls = {}
        png_bounds = tuple(gdf_aoi.to_crs(4326).total_bounds)
        leaflet_bounds = [[png_bounds[1], png_bounds[0]],
                          [png_bounds[3], png_bounds[2]]]

        heat_path = os.path.join(results_dir, "pv_probability.png")
        pu.save_png_heatmap(proba, heat_path)
        layer_urls['pv_probability'] = f"/{heat_path}"

        rgb_path = os.path.join(results_dir, "sentinel_true_color.png")
        pu.save_png_rgb(band_arrays, rgb_path)
        layer_urls['sentinel_true_color'] = f"/{rgb_path}"

        # 8. Uncertainty Estimation (DI/LPD)
        if calculate_uncertainty:
            # Save prediction data for R
            pd.DataFrame(X, columns=feature_names).to_csv(os.path.join(
                results_dir, "prediction_features.csv"), index=False)

            with rasterio.open(os.path.join(results_dir, "building_mask.tif"), 'w', driver='GTiff',
                               height=H, width=W, count=1, dtype='uint8', crs=dst_crs, transform=transform) as ds:
                ds.write(building_mask.astype('uint8'), 1)

            with open(os.path.join(results_dir, "spatial_meta.json"), "w") as f:
                json.dump({"height": H, "width": W, "crs_wkt": dst_crs.to_wkt(
                ), "transform": transform.to_gdal()}, f)

            # Fix for KeyError: 'export'
            source_csv = CFG.get('export', {}).get(
                'csv_path') or CFG['prediction'].get('original_training_data_csv')

            if source_csv and os.path.exists(source_csv):
                shutil.copyfile(source_csv, os.path.join(
                    results_dir, "training_data.csv"))
                if pu.run_r_script(r_script_path, os.path.abspath(results_dir)):
                    for layer in ["di", "lpd"]:
                        tif = os.path.join(results_dir, f"aoi_{layer}.tif")
                        if os.path.exists(tif):
                            png = os.path.join(results_dir, f"{layer}.png")
                            pu.save_geotiff_as_colored_png(
                                tif, png, "RdYlGn" if layer == "lpd" else "RdYlGn_r", layer)
                            layer_urls[layer] = f"/{png}"

        return jsonify({
            "status": "success", "layers": layer_urls, "bounds": leaflet_bounds,
            "classification": {"score": round(float(np.nanpercentile(proba, 90)), 3) if not np.all(np.isnan(proba)) else 0}
        })

    except Exception as e:
        LOGGER.error(f"Predict error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
