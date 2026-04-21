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

import prediction_utils as pu

app = flask.Flask(__name__)
pu.setup_logging()
LOGGER = logging.getLogger("pv-flask-app")

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

THRESHOLD = 0.715
MODEL = None

R_MODEL_PATH = os.path.abspath("data/artifacts/rf/model_final.rds")
if not os.path.exists(R_MODEL_PATH):
    LOGGER.warning(
        f"WARNING: R model not found at {R_MODEL_PATH}. Prediction will fail.")
else:
    LOGGER.info(f"App initialized to use R model at: {R_MODEL_PATH}")


@app.route('/')
def index():
    frontend_config = {
        'planet_api_key': CFG.get('api_keys', {}).get('planet_labs', "")
    }
    return render_template('index.html', config=frontend_config)


@app.route('/api/find_building', methods=['POST'])
def find_building():
    """
    Finds the closest building to a clicked point.
    """
    try:
        data = request.json
        lon, lat = data['lon'], data['lat']

        clicked_point_wkt = f"POINT({lon} {lat})"
        tbl = db_cfg['footprints_table']
        geom_col = db_cfg['geometry_column']


        with ENGINE.connect() as conn:

            srid_query = text(
                f"SELECT ST_SRID(t.{geom_col}) FROM {tbl} t LIMIT 1")
            srid_result = conn.execute(srid_query)
            native_srid = srid_result.fetchone()[0]

            if not native_srid:
                raise Exception(f"Could not determine SRID for table {tbl}")

            query = text(f"""
                SELECT 
                    json_build_object('type', 'Feature', 'geometry', ST_AsGeoJSON(ST_Transform(t.{geom_col}, 4326))::json) as geojson
                FROM 
                    {tbl} t
                ORDER BY 
                    t.{geom_col} <-> ST_Transform(ST_GeomFromText(:point_wkt, 4326), :native_srid)
                LIMIT 1;
            """)

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
    Runs prediction and yields progress updates as a stream of JSON chunks.
    """
    data = request.json
    building_geojson = data['building_geojson']
    include_aoa = data.get('include_aoa', False)
    building_geom = shape(building_geojson['geometry'])

    def generate(building_geojson, include_aoa, building_geom):
        try:
            yield json.dumps({"status": "progress", "message": "Fetching Sentinel-2 data"}) + "\n"

            session_id = str(uuid.uuid4())
            results_dir = os.path.join('static', 'results', session_id)
            os.makedirs(results_dir, exist_ok=True)
            r_script_dir = os.path.abspath(results_dir)
            r_script_path = os.path.abspath("scripts/predict.R")
            pred_cfg = CFG['prediction']
            target_epsg = 3857

            gdf_aoi, bbox_target, bbox_4326 = pu.get_aoi_from_building_geom(
                building_geom, target_epsg, pred_cfg['buffer_m']
            )

            stac_item = pu.stac_search_one(pred_cfg, bbox_4326)

            stac_bands = ["B02", "B03", "B04", "B05",
                          "B06", "B07", "B08", "B8A", "B11", "B12"]
            band_arrays, transform, dst_crs = pu.read_bands_to_grid(
                stac_item, stac_bands, bbox_target, target_epsg, resolution=10.0
            )
            H, W = next(iter(band_arrays.values())).shape

            building_gdf = gpd.GeoDataFrame(
                [{'geometry': building_geom}], crs=4326)
            building_mask = pu.create_building_mask(
                building_gdf, transform, (H, W), dst_crs)

            X, feature_names = pu.assemble_features(
                None, band_arrays, feature_list=stac_bands)
            pd.DataFrame(X, columns=feature_names).to_csv(
                os.path.join(r_script_dir, "prediction_features.csv"), index=False
            )

            yield json.dumps({"status": "progress", "message": "Running prediction"}) + "\n"

            if include_aoa:
                yield json.dumps({"status": "progress", "message": "Calculating DI"}) + "\n"

            success = pu.run_r_script(
                r_script_path, r_script_dir, run_aoa=include_aoa)
            if not success:
                raise Exception("R script failed to generate predictions.")

            if include_aoa:
                yield json.dumps({"status": "progress", "message": "Calculating LPD"}) + "\n"

            yield json.dumps({"status": "progress", "message": "Generating visualization layers..."}) + "\n"

            preds_path = os.path.join(r_script_dir, "prediction_results.csv")
            r_preds = pd.read_csv(preds_path)
            pv_col = [c for c in r_preds.columns if 'PV' in c or 'X1' in c][0]
            proba = r_preds[pv_col].values.reshape((H, W))
            proba[building_mask == 0] = np.nan

            valid_pixels = proba[building_mask == 1]
            valid_pixels = valid_pixels[~np.isnan(valid_pixels)]
            pv_score = float(np.percentile(valid_pixels, 90)
                             ) if len(valid_pixels) > 0 else 0.0
            is_pv = pv_score >= THRESHOLD

            layer_urls = {}
            png_bounds = tuple(gpd.GeoSeries(
                gdf_aoi, crs=target_epsg).to_crs(4326).total_bounds)
            png_bounds_leaflet = [[png_bounds[1], png_bounds[0]], [
                png_bounds[3], png_bounds[2]]]

            pu.save_png_heatmap(proba, os.path.join(
                results_dir, "pv_probability.png"), cmap_name="RdYlGn")
            layer_urls['pv_probability'] = f"/{results_dir}/pv_probability.png"

            pu.save_png_rgb(band_arrays, os.path.join(
                results_dir, "sentinel_true_color.png"))
            layer_urls['sentinel_true_color'] = f"/{results_dir}/sentinel_true_color.png"

            if include_aoa:
                metrics_path = os.path.join(r_script_dir, "spatial_metrics.csv")
                if os.path.exists(metrics_path):
                    r_metrics = pd.read_csv(metrics_path)

                    def get_masked_stats(data_flat):
                        data_2d = data_flat.reshape((H, W))
                        data_2d[building_mask == 0] = np.nan
                        valid = data_2d[building_mask == 1]
                        valid = valid[~np.isnan(valid)]
                        if len(valid) > 0:
                            return data_2d, float(np.min(valid)), float(np.max(valid))
                        return data_2d, 0.0, 1.0

                    di_data, di_min, di_max = get_masked_stats(r_metrics['DI'].values.astype(float))
                    pu.save_png_heatmap(di_data, os.path.join(
                        results_dir, "dissimilarity_index.png"), cmap_name="RdYlGn_r", vmin=di_min, vmax=di_max)
                    layer_urls['dissimilarity_index_(DI)'] = f"/{results_dir}/dissimilarity_index.png"

                    lpd_data, lpd_min, lpd_max = get_masked_stats(r_metrics['LPD'].values.astype(float))
                    pu.save_png_heatmap(lpd_data, os.path.join(
                        results_dir, "local_data_density.png"), cmap_name="RdYlGn_r", vmin=lpd_min, vmax=lpd_max)
                    layer_urls['local_data_density_(LPD)'] = f"/{results_dir}/local_data_density.png"

            final_result = {
                "status": "success",
                "layers": layer_urls,
                "bounds": png_bounds_leaflet,
                "classification": {
                    "is_pv": is_pv,
                    "score": round(pv_score, 3),
                    "text": "PV Detected" if is_pv else "No PV Detected"
                }
            }
            yield json.dumps(final_result) + "\n"

        except Exception as e:
            LOGGER.error(f"Error in /api/predict: {e}", exc_info=True)
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    return flask.Response(generate(building_geojson, include_aoa, building_geom), mimetype='application/x-ndjson')


if __name__ == '__main__':
    app.run(host=CFG['server']['host'], port=CFG['server']['port'], debug=True)
