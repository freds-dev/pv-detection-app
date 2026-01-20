import pandas as pd
import numpy as np
import json
import requests
from sqlalchemy import create_engine, text
import yaml
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

# --- Settings ---
API_URL = "http://localhost:5000/api/predict"
OUTPUT_PATH = "data/artifacts/validation_results.csv"
TRAINING_DATA_PATH = "data/training/training.csv"
META_PATH = "data/artifacts/validate_binary_classification/test_metadata.json"
MAX_WORKERS = 4
RETRIES = 2

# Setup Database
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

DB_URL = f"postgresql+psycopg2://{CFG['database']['user']}:{CFG['database']['password']}@{CFG['database']['host']}:{CFG['database']['port']}/{CFG['database']['dbname']}"
ENGINE = create_engine(DB_URL)


def get_building_data(lon, lat):
    """Fetches GeoJSON and Area (sqm) from PostGIS using coordinates."""
    tbl = CFG['database']['footprints_table']
    geom_col = CFG['database']['geometry_column']

    # We transform the CSV point (3857) to the native SRID of your database table
    query = text(f"""
        SELECT 
            ST_AsGeoJSON(ST_Transform({geom_col}, 4326)),
            ST_Area(ST_Transform({geom_col}, 3857)) as area_sqm
        FROM {tbl} 
        WHERE ST_Intersects(
            {geom_col}, 
            ST_Transform(
                ST_SetSRID(ST_Point(:lon, :lat), 3857), 
                (SELECT ST_SRID({geom_col}) FROM {tbl} LIMIT 1)
            )
        ) 
        LIMIT 1
    """)

    with ENGINE.connect() as conn:
        res = conn.execute(query, {"lon": lon, "lat": lat}).fetchone()
        if res:
            return json.loads(res[0]), res[1]
        return None, None


def validate_single_row(row):
    poly_id = str(row['parent_poly_id'])
    geojson, area_sqm = get_building_data(row['lon'], row['lat'])

    # If the spatial query fails, we can't predict.
    if not geojson:
        # tqdm.write(f"🔍 Skip: No building found in DB for {poly_id} at {row['lon']}, {row['lat']}")
        return {"skip": True}

    for attempt in range(RETRIES):
        try:
            response = requests.post(API_URL, json={"building_geojson": {
                                     "geometry": geojson}}, timeout=120)
            if response.status_code == 200:
                data = response.json()
                return {
                    "poly_id": poly_id,
                    "ground_truth": int(row['label']),
                    "prediction": 1 if data['classification']['is_pv'] else 0,
                    "score": data['classification']['score'],
                    "area_sqm": area_sqm
                }
        except Exception:
            time.sleep(2)
    return {"error": f"Failed API call for {poly_id}"}


def analyze_and_plot(df):
    """Calculates global stats and Recall by Size."""
    if df.empty:
        print("❌ No successful results to analyze.")
        return

    print(f"\n📊 Analyzing {len(df)} successful validations...")

    y_true = df['ground_truth']
    y_score = df['score']

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    accuracies = [accuracy_score(y_true, y_score >= t) for t in thresholds]
    best_acc_threshold = thresholds[np.argmax(accuracies)]

    print("═"*50)
    print(f"AUC: {roc_auc:.4f} | Optimal Threshold: {best_acc_threshold:.2f}")
    print("═"*50)

    bins = [0, 100, 200, 500, 1000, 5000, 1000000]
    labels = ['<100', '100-200', '200-500', '500-1k', '1k-5k', '>5k']
    df['size_bin'] = pd.cut(df['area_sqm'], bins=bins, labels=labels)

    size_stats = df.groupby('size_bin', observed=True).apply(
        lambda g: pd.Series({
            'recall': (((g['score'] >= best_acc_threshold).astype(int) == 1) & (g['ground_truth'] == 1)).sum() / (g['ground_truth'] == 1).sum() if (g['ground_truth'] == 1).sum() > 0 else np.nan,
            'samples': len(g)
        })
    )
    print(size_stats)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    y_pred = (y_score >= best_acc_threshold).astype(int)
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True,
                fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Confusion Matrix")

    sns.barplot(x=size_stats.index,
                y=size_stats['recall'], palette="viridis", ax=axes[1])
    axes[1].set_title("Recall by Building Size (sqm)")
    axes[1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.show()


def run_validation():
    # 1. Load Metadata
    if not os.path.exists(META_PATH):
        print(f"❌ Error: {META_PATH} not found.")
        return
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    test_ids = set(str(x) for x in meta['test_polygons'])

    # 2. Setup Resume
    processed_ids = set()
    all_results = []
    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0:
        try:
            df_existing = pd.read_csv(OUTPUT_PATH)
            if not df_existing.empty:
                all_results = df_existing.to_dict('records')
                processed_ids = set(str(x)
                                    for x in df_existing['poly_id'].unique())
                print(f"🔄 Resuming: {len(processed_ids)} already processed.")
        except Exception:
            os.remove(OUTPUT_PATH)

    # 3. Load Training Data
    df_train = pd.read_csv(TRAINING_DATA_PATH, on_bad_lines='skip')
    df_train['parent_poly_id'] = df_train['parent_poly_id'].astype(str)

    df_test = df_train[df_train['parent_poly_id'].isin(test_ids)].copy()
    df_test = df_test.drop_duplicates(subset=['parent_poly_id'])
    df_todo = df_test[~df_test['parent_poly_id'].isin(processed_ids)].copy()

    print(
        f"📋 Total Test Polygons: {len(df_test)} | Remaining to process: {len(df_todo)}")

    if len(df_todo) == 0:
        if all_results:
            analyze_and_plot(pd.DataFrame(all_results))
        return

    # 4. Processing Loop
    skipped_count = 0
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            tasks = [row for _, row in df_todo.iterrows()]
            futures = {executor.submit(
                validate_single_row, t): t for t in tasks}

            with tqdm(total=len(tasks), desc="Validating") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        if res.get("skip"):
                            skipped_count += 1
                        elif "error" in res:
                            tqdm.write(f"⚠️ {res['error']}")
                        else:
                            all_results.append(res)
                            if len(all_results) % 20 == 0:
                                pd.DataFrame(all_results).to_csv(
                                    OUTPUT_PATH, index=False)
                    pbar.update(1)
    finally:
        print(
            f"\n✅ Finished. Skipped {skipped_count} buildings (not found in DB).")
        if all_results:
            df_final = pd.DataFrame(all_results)
            df_final.to_csv(OUTPUT_PATH, index=False)
            analyze_and_plot(df_final)


if __name__ == "__main__":
    run_validation()
