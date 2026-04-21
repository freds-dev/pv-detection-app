import geopandas as gpd
import yaml
import os
import logging
from sqlalchemy import create_engine
from geoalchemy2 import Geometry

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("preprocess-buildings")

try:
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f) or {}
    PRED_CONFIG = CONFIG.get("prediction", {})
    BUILDING_GPKG = PRED_CONFIG.get(
        "building_footprints_gpkg", "data/ground_truth/all_building_footprints.gpkg")
    TARGET_EPSG = CONFIG.get("target_epsg", 3857)
    MIN_AREA_SQM = PRED_CONFIG.get("min_building_area_sqm", 0)

    db_dir = os.path.dirname(BUILDING_GPKG)
    DB_PATH = os.path.join(db_dir, "buildings_index.sqlite")

except FileNotFoundError:
    LOGGER.error("FATAL: config.yaml not found.")
    exit(1)
except Exception as e:
    LOGGER.error(f"FATAL: Error loading config: {e}")
    exit(1)


def create_spatial_database():
    """
    One-time script to:
    1. Load the massive building GeoPackage.
    2. Filter it by minimum area.
    3. Save the result to a SpatiaLite database with a spatial index.
    """
    if not os.path.exists(BUILDING_GPKG):
        LOGGER.error(
            f"FATAL: Building footprints file not found at {BUILDING_GPKG}")
        return

    LOGGER.info(f"Loading building footprints from {BUILDING_GPKG}...")
    try:
        gdf_all = gpd.read_file(BUILDING_GPKG)
        LOGGER.info(f"Loaded {len(gdf_all)} total buildings.")
    except Exception as e:
        LOGGER.error(f"Failed to read GeoPackage: {e}")
        return

    if MIN_AREA_SQM > 0:
        LOGGER.info(
            f"Filtering buildings by min_area_sqm >= {MIN_AREA_SQM}...")
        try:
            gdf_projected = gdf_all.to_crs(f"EPSG:{TARGET_EPSG}")

            gdf_projected['area_sqm'] = gdf_projected.geometry.area

            filtered_indices = gdf_projected[gdf_projected['area_sqm']
                                             >= MIN_AREA_SQM].index

            gdf_filtered = gdf_all.loc[filtered_indices]
            LOGGER.info(f"Filtered down to {len(gdf_filtered)} buildings.")
        except Exception as e:
            LOGGER.error(f"Error during filtering: {e}. Using all buildings.")
            gdf_filtered = gdf_all
    else:
        LOGGER.info("No minimum area filter set, using all buildings.")
        gdf_filtered = gdf_all

    if gdf_filtered.empty:
        LOGGER.error("No buildings left after filtering. Aborting.")
        return

    LOGGER.info("Re-projecting filtered buildings to EPSG:4326...")
    gdf_filtered_4326 = gdf_filtered.to_crs("EPSG:4326")

    LOGGER.info(f"Creating SpatiaLite database at {DB_PATH}...")
    engine_str = f"sqlite:///{DB_PATH}"
    engine = create_engine(engine_str)

    try:
        gdf_filtered_4326.to_sql(
            'buildings',
            con=engine,
            if_exists='replace',
            index=True,  
            index_label='fid',  
            dtype={'geometry': Geometry('POLYGON', srid=4326)}
        )

        LOGGER.info("Database table 'buildings' created successfully.")

        LOGGER.info("Creating spatial index...")
        with engine.connect() as conn:
            conn.exec_driver_sql(
                "SELECT CreateSpatialIndex('buildings', 'geometry');")

        LOGGER.info("Spatial index created successfully.")

    except Exception as e:
        LOGGER.error(f"Failed to write to SpatiaLite database: {e}")
        LOGGER.error(
            "Please ensure you have 'sqlalchemy', 'geoalchemy2', and 'pysqlite3' (or 'spatialite') installed.")
        return

    LOGGER.info(f"--- SUCCESS ---")
    LOGGER.info(
        f"Pre-processing complete. Your database is ready at {DB_PATH}")
    LOGGER.info(f"You can now run 'python app.py'")


if __name__ == "__main__":
    create_spatial_database()
