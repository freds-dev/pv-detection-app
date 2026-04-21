# Interactive PV Prediction Map

Application for industiral rooftop PV prediction using Sentinel-2 imagery and Random Forest models.

## Project Structure
- `app.py`: Flask server & API.
- `prediction_utils.py`: Imagery processing & visualization.
- `scripts/predict.R`: R script for DI/LPD calculation via `CAST`.
- `config.yaml`: System configuration (DB, Model paths, STAC).
- `data/`: Model artifacts and training data.

## Quick Start Setup

### 1. Database (PostGIS)
Run the database container:
```bash
docker run --name pv-gis-db -e POSTGRES_USER=postgres -e POSTGRES_DB=gisdb -e POSTGRES_PASSWORD=postgres -p 5433:5432 -d ankane/pgvector
docker exec -it pv-gis-db psql -U postgres -d gisdb -c "CREATE EXTENSION postgis;"
```
Load your footprints (GeoPackage):
```bash
ogr2ogr -f "PostgreSQL" PG:"host=localhost port=5433 dbname=gisdb user=postgres password=postgres" "footprints.gpkg" -nln building_footprints -lco GEOMETRY_NAME=geom -nlt PROMOTE_TO_MULTI -overwrite
docker exec -it pv-gis-db psql -U postgres -d gisdb -c "CREATE INDEX building_footprints_geom_idx ON building_footprints USING GIST (geom); ALTER SYSTEM SET jit = off; SELECT pg_reload_conf();"
```

### 2. Environments
**Python:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
**R:**
```R
install.packages(c("CAST", "ranger", "jsonlite", "caret", "terra"))
```

### 3. Configuration
Create `config.yaml`:
```yaml
server:
  host: "0.0.0.0"
  port: 5000
database:
  host: "localhost"
  port: 5433
  dbname: "gisdb"
  user: "postgres"
  password: "postgres"
  footprints_table: "building_footprints" 
  geometry_column: "geom" 
prediction:
  model_path: "data/artifacts/rf/model_final.rds"
  original_training_data_csv: "data/training/training.csv"
  buffer_m: 100
  stac_api: "https://planetarycomputer.microsoft.com/api/stac/v1"
  collection: "sentinel-2-l2a"
  date_range: "2025-01-01/2025-09-10"
  cloud_cover: 20
```

## Running the App
1. `docker start pv-gis-db`
2. `source venv/bin/activate`
3. `ulimit -s unlimited` (macOS/Linux)
4. `python app.py`
5. Access at **http://localhost:5000**
