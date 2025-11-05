# Interactive PV Prediction Map

## 1. Description

This is a full-stack web application for interactively running a machine learning model for rooftop photovoltaic (PV) suitability.

A user can browse an interactive map, click any location, and the application will find the nearest building footprint from a spatial database. The user can then trigger a prediction for that specific building. The backend fetches Sentinel-2 satellite imagery for the area, runs a prediction model, and calculates a Dissimilarity Index (DI) and Local Point Density (LPD) to assess prediction reliability. The resulting layers are then overlaid on the map for the user to review.

## 2. Features

* **Interactive Map Interface:** Built with Leaflet.js.
* **Basemaps:** Includes Google Satellite, OpenStreetMap, a light basemap, and Planet Labs (if API key is provided).
* **Geocoding:** Search for locations via a text-based search bar.
* **Database-Backed Footprints:** Clicking the map queries a PostGIS database to find the nearest building footprint, ensuring high performance even with national-scale datasets.
* **On-Demand Prediction:** Fetches satellite data from Planetary Computer (STAC) for a buffered AOI around the selected building.
* **Model Inference:** Runs a pre-trained `scikit-learn` model to generate a PV probability heatmap.
* **Reliability Metrics:** Executes an R script (`CAST` package) to generate Dissimilarity Index (DI) and Local Point Density (LPD) layers, providing context on how "out-of-sample" the prediction area is.
* **Dynamic UI:** Includes a loading overlay during prediction and request-cancellation logic to prevent race conditions.

## 3. Architecture Overview

This application operates on a client-server model.



* **Frontend (Client):** A single HTML/CSS/JavaScript file (`templates/index.html`) that runs in the user's browser. It uses **Leaflet.js** for mapping and `fetch` to communicate with the backend.
* **Backend (Server):** A **Flask** (`app.py`) server that handles API requests.
* **Database:** A **PostGIS** database running in a **Docker** container. It stores all building footprints and uses a spatial index for fast nearest-neighbor queries.
* **Prediction Model:** A pre-trained `.joblib` model (e.g., a `scikit-learn` Random Forest) that is loaded by Flask for inference.
* **DI/LPD Model:** An R script (`scripts/calculate_di.R`) that is executed via `subprocess` by Flask. It trains a temporary `ranger` model to calculate the Area of Applicability (AOA).

## 4. Project Structure

```
/your-project-name
|
|-- app.py                 # The main Flask web server
|-- prediction_utils.py    # Core prediction logic (STAC, PNG saving, etc.)
|-- config.yaml            # Your local configuration (passwords, paths)
|-- requirements.txt       # Python package list
|
|-- /templates
|   |-- index.html         # The frontend (map interface)
|
|-- /static
|   |-- /results/          # (Created automatically) Stores temporary prediction PNGs
|
|-- /scripts
|   |-- calculate_di.R     # R script for DI/LPD calculation
|
|-- /data
|   |-- /artifacts
|   |   |-- random_forest.joblib  # (User must add this)
|   |-- /training
|   |   |-- training.csv          # (User must add this)
|
|-- README.md              # This file
```

---

## 5. Installation and Setup

This is a comprehensive, one-time setup guide to get the application running from scratch.

### Prerequisites

Before you begin, ensure the following software is installed on your system:

* **Git:** For cloning the repository.
* **Docker Desktop:** For running the PostGIS database container.
* **Python 3.10+:** For the Flask web server.
* **R:** For running the `Rscript` command and DI/LPD analysis.
* **GDAL (`ogr2ogr`):** A command-line tool for loading spatial data.
    * *macOS (Homebrew):* `brew install gdal`
    * *Conda:* `conda install -c conda-forge gdal`
    * *Linux (apt):* `sudo apt-get install gdal-bin`

### Step 1: Clone Repository

Clone this repository to your local machine:
```bash
git clone <your-repository-url>
cd <your-project-name>
```

### Step 2: Database Setup (PostGIS)

This application requires a PostGIS database to store and query building footprints.

**2.1. Run the Docker Container**
This command downloads and runs a multi-architecture PostGIS-enabled database container. It maps your local port `5433` to the container's port `5432` to avoid common port conflicts.

```bash
docker run --name pv-gis-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=gisdb \
  -e POSTGRES_PASSWORD=your_secret_password \
  -p 5433:5432 \
  -d ankane/pgvector
```
**Note:** Remember the `your_secret_password` you choose. You will need it for your `config.yaml` file.

**2.2. Install PostGIS Extension (Inside the Container)**
The base image requires the PostGIS extension to be manually installed.
```bash
# Get a root shell inside the running container
docker exec -it -u root pv-gis-db bash

# (You are now inside the container)
apt-get update
apt-get install -y postgresql-15-postgis-3
exit
```

**2.3. Enable the PostGIS Extension (SQL)**
This command activates the PostGIS extension within the `gisdb` database.
```bash
docker exec -it pv-gis-db psql -U postgres -d gisdb -c "CREATE EXTENSION postgis;"
```

**2.4. Load Building Footprint Data**
Use the `ogr2ogr` tool to load your building footprints GeoPackage (`.gpkg`) file into the database. This step may take a long time.

**Replace `path/to/your/all_building_footprints.gpkg` and `your_secret_password`** with your actual values.
```bash
ogr2ogr -f "PostgreSQL" \
  PG:"host=localhost port=5433 dbname=gisdb user=postgres password=your_secret_password" \
  "path/to/your/all_building_footprints.gpkg" \
  -nln building_footprints \
  -lco GEOMETRY_NAME=geom \
  -lco FID=id \
  -nlt PROMOTE_TO_MULTI \
  -overwrite
```

**2.5. Create Spatial Index (CRITICAL)**
This step is essential for application performance. It indexes the geometry column for fast spatial queries. This will also take several minutes.
```bash
docker exec -it pv-gis-db psql -U postgres -d gisdb -c "CREATE INDEX building_footprints_geom_idx ON building_footprints USING GIST (geom);"
```

**2.6. Disable JIT Compiler**
This step prevents a low-level PostGIS/JIT compatibility crash when running complex spatial queries.
```bash
docker exec -it pv-gis-db psql -U postgres -d gisdb -c "ALTER SYSTEM SET jit = off;"
docker exec -it pv-gis-db psql -U postgres -d gisdb -c "SELECT pg_reload_conf();"
```
The database is now fully configured and running.

### Step 3: Environment Configuration

**3.1. Create `config.yaml`**
In the project's root directory, create a new file named `config.yaml`. Copy the contents from the `config.yaml.example` in the Appendix (Section 7) into this new file.

**3.2. Edit `config.yaml`**
Open your new `config.yaml` and:
1.  Set `database.password` to the `your_secret_password` you chose in Step 2.1.
2.  Set `prediction.model_path` to the path where you will place your trained `.joblib` model file (e.g., `data/artifacts/random_forest.joblib`).
3.  Set `prediction.original_training_data_csv` to the path where you will place your full training `.csv` file (e.g., `data/training/training.csv`).

**3.3. Add Model and Data Files**
Manually copy your trained model (`.joblib`) and your full training data (`.csv`) into the locations you specified in `config.yaml`. If the `data/artifacts` or `data/training` directories do not exist, create them.

### Step 4: R Environment Setup

**4.1. Install R Packages**
The `calculate_di.R` script requires several R packages. Open an R shell (by typing `R` in your terminal) and run the following command:
```R
install.packages(c("here", "CAST", "ranger", "readr", "terra", "caret", "jsonlite", "ggplot2", "lattice"))
```

**4.2. Configure R Script**
You *must* edit one line in `scripts/calculate_di.R` to tell the script which column in your `training.csv` file contains the actual labels.

* Open `scripts/calculate_di.R`.
* Find the line (around line 51): `outcome_col <- "label"`
* Change `"label"` to the actual name of your label column (e.g., `"pv_label"`, `"class"`, `y`, etc.).

### Step 5: Python Environment Setup

**5.1. Create Virtual Environment**
From the project's root directory:
```bash
python3 -m venv venv
```

**5.2. Activate Environment**
```bash
source venv/bin/activate
```
*(On Windows, use: `venv\Scripts\activate`)*

**5.3. Install Python Packages**
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 6. Running the Application

After completing the one-time setup, follow these steps to run the server.

**Step 1: Start the Database Container**
If your computer has been restarted, your Docker container is stopped. Start it:
```bash
docker start pv-gis-db
```

**Step 2: Activate the Python Environment**
Navigate to the project directory and activate the environment:
```bash
cd /path/to/your/prediction-app
source venv/bin/activate
```

**Step 3: Set Memory Limit (macOS/Linux Only)**
To prevent the R script from crashing on large computations, increase the shell's stack size limit:
```bash
ulimit -s unlimited
```

**Step 4: Run the Flask Server**
```bash
python app.py
```

**Step 5: Access the Application**
Open your web browser and navigate to:
**[http://localhost:5000](http://localhost:5000)**

---

## 7. Appendix: Configuration Files

### `requirements.txt`
```
flask
geopandas
rasterio
folium
branca
joblib
pandas
pyyaml
numpy
planetary-computer
pystac-client
matplotlib
Pillow
SQLAlchemy
GeoAlchemy2
psycopg2-binary
scikit-learn
```

### `config.yaml.example`
*(Copy this to `config.yaml` and edit it)*
```yaml
# =========================
# Web Application Settings
# =========================
server:
  host: "0.0.0.0"
  port: 5000

# =========================
# Database Settings (PostGIS)
# =========================
database:
  host: "localhost"
  port: 5433 # Use 5433 to avoid conflicts
  dbname: "gisdb"
  user: "postgres"
  password: "your_secret_password" # Change this
  footprints_table: "building_footprints" 
  geometry_column: "geom" 

# =========================
# Map & API Keys
# =========================
api_keys:
  # (Optional)
  planet_labs: "YOUR_PLANET_API_KEY_HERE" 

# =========================
# Prediction Model Settings
# =========================
prediction:
  # Path to your trained model file
  model_path: "data/artifacts/random_forest.joblib"

  # Path to the *original* training data CSV (features + labels)
  # The R script needs this for DI/LPD comparison.
  original_training_data_csv: "data/training/training.csv"

  # Buffer (in meters) around the selected building to run prediction
  buffer_m: 100

  # Minimum building area (sq. meters) to be selectable
  min_building_area_sqm: 50

  # STAC settings for Sentinel data
  stac_api: "[https://planetarycomputer.microsoft.com/api/stac/v1](https://planetarycomputer.microsoft.com/api/stac/v1)"
  collection: "sentinel-2-l2a"
  date_range: "2025-01-01/2025-09-10"
  cloud_cover: 20
```