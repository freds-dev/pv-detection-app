import geopandas as gpd
import yaml
import os
import logging
from sqlalchemy import create_engine
from geoalchemy2 import Geometry

# --- Setup ---
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("preprocess-buildings")

# --- Configuration ---
try:
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f) or {}
    PRED_CONFIG = CONFIG.get("prediction", {})
    BUILDING_GPKG = PRED_CONFIG.get(
        "building_footprints_gpkg", "data/ground_truth/all_building_footprints.gpkg")
    TARGET_EPSG = CONFIG.get("target_epsg", 3857)
    MIN_AREA_SQM = PRED_CONFIG.get("min_building_area_sqm", 0)

    # Define the output path for the new database
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

    # --- 1. Filter by Area (Your suggestion) ---
    if MIN_AREA_SQM > 0:
        LOGGER.info(
            f"Filtering buildings by min_area_sqm >= {MIN_AREA_SQM}...")
        try:
            # Reproject to a meter-based CRS to accurately calculate area
            gdf_projected = gdf_all.to_crs(f"EPSG:{TARGET_EPSG}")

            # Calculate area in sqm
            gdf_projected['area_sqm'] = gdf_projected.geometry.area

            # Get the indices of buildings that meet the criteria
            filtered_indices = gdf_projected[gdf_projected['area_sqm']
                                             >= MIN_AREA_SQM].index

            # Filter the original GDF using the indices
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

    # --- 2. Reproject for SpatiaLite ---
    # SpatiaLite and GeoJSON work best with EPSG:4326 (Lat/Lon)
    LOGGER.info("Re-projecting filtered buildings to EPSG:4326...")
    gdf_filtered_4326 = gdf_filtered.to_crs("EPSG:4326")

    # --- 3. Save to SpatiaLite Database ---
    LOGGER.info(f"Creating SpatiaLite database at {DB_PATH}...")
    # 'spatialite:///path/to/db.sqlite'
    engine_str = f"sqlite:///{DB_PATH}"
    engine = create_engine(engine_str)

    try:
        # This is the key step: save the GeoDataFrame directly to SQL
        # This will automatically create a 'buildings' table
        gdf_filtered_4326.to_sql(
            'buildings',
            con=engine,
            if_exists='replace',
            index=True,  # Use the original GDF index as the primary key
            index_label='fid',  # Name the index column 'fid'
            dtype={'geometry': Geometry('POLYGON', srid=4326)}
        )

        LOGGER.info("Database table 'buildings' created successfully.")

        # --- 4. Create Spatial Index ---
        # This is CRITICAL for fast queries
        LOGGER.info("Creating spatial index...")
        # We must use a separate connection to execute raw SQL
        with engine.connect() as conn:
            # SpatiaLite's function to create a spatial index (R-Tree)
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
