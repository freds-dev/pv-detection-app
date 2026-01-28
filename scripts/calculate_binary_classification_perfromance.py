import pandas as pd
import numpy as np
import json
import geopandas as gpd
from sqlalchemy import create_engine, text
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

# --- Settings ---
RESULTS_PATH = "data/artifacts/validation_results.csv"
TRAINING_PATH = "data/training/training.csv"
ERROR_GPKG_PATH = "data/artifacts/worst_guesses.gpkg"
PLOT_EXPORT_PATH = "output/plots"

# Setup
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

DB_URL = f"postgresql+psycopg2://{CFG['database']['user']}:{CFG['database']['password']}@{CFG['database']['host']}:{CFG['database']['port']}/{CFG['database']['dbname']}"
ENGINE = create_engine(DB_URL)


def export_worst_guesses_spatial(df, threshold=0.71):
    """Identifies both high-confidence False Positives and missed False Negatives."""
    print("🔗 Merging results with training coordinates...")
    df_train = pd.read_csv(TRAINING_PATH, usecols=[
                           'lon', 'lat', 'parent_poly_id'])
    df_train['parent_poly_id'] = df_train['parent_poly_id'].astype(str)
    df['poly_id'] = df['poly_id'].astype(str)

    df = df.merge(df_train, left_on='poly_id',
                  right_on='parent_poly_id', how='left')

    fps = df[(df['ground_truth'] == 0) & (df['score'] >= threshold)].copy()
    fps['error_type'] = 'False Positive'

    fns = df[(df['ground_truth'] == 1) & (df['score'] < threshold)].copy()
    fns['error_type'] = 'False Negative'

    worst_df = pd.concat([fps, fns])

    if worst_df.empty:
        print("⚠️ No errors found matching the threshold criteria.")
        return

    print(
        f"🛰️ Requesting {len(fps)} FPs and {len(fns)} FNs from the database...")

    tbl = CFG['database']['footprints_table']
    geom_col = CFG['database']['geometry_column']
    features = []

    with ENGINE.connect() as conn:
        for _, row in worst_df.iterrows():
            query = text(f"""
                SELECT ST_AsGeoJSON(ST_Transform({geom_col}, 4326))
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
            res = conn.execute(
                query, {"lon": row['lon'], "lat": row['lat']}).fetchone()

            if res:
                geom_json = json.loads(res[0])
                feat = {
                    'type': 'Feature',
                    'geometry': geom_json,
                    'properties': {
                        'poly_id': row['poly_id'],
                        'GT': int(row['ground_truth']),
                        'Score': round(float(row['score']), 4),
                        'Area': int(row['area_sqm']),
                        'Type': row['error_type']
                    }
                }
                features.append(feat)

    if features:
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        os.makedirs(os.path.dirname(ERROR_GPKG_PATH), exist_ok=True)
        gdf.to_file(ERROR_GPKG_PATH, driver="GPKG")
        print(
            f"✅ Success! {len(gdf)} total error polygons exported to: {ERROR_GPKG_PATH}")


def analyze_and_plot(df):
    """Statistical and visual analysis."""
    if 'ground_truth' not in df.columns and 'label' in df.columns:
        df = df.rename(columns={'label': 'ground_truth'})

    y_true, y_score = df['ground_truth'], df['score']
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    accuracies = [accuracy_score(y_true, y_score >= t) for t in thresholds]
    best_acc_idx = np.argmax(accuracies)
    best_acc_thr = thresholds[best_acc_idx]
    best_fpr, best_tpr = fpr[best_acc_idx], tpr[best_acc_idx]

    print("\n" + "═"*60)
    print(f"📊 PERFORMANCE REPORT")
    print(f"AUC: {roc_auc:.4f} | Optimal Threshold (Max Acc): {best_acc_thr:.4f}")
    print("═"*60)

    # Binning by size
    bins = [0, 100, 200, 500, 1000, 5000, 1000000]
    labels = ['<100', '100-200', '200-500', '500-1k', '1k-5k', '>5k']
    df['size_bin'] = pd.cut(df['area_sqm'], bins=bins, labels=labels)

    size_stats = []
    for label in labels:
        g = df[df['size_bin'] == label]
        if len(g) > 0:
            relevant = (g['ground_truth'] == 1).sum()
            rec = (((g['score'] >= 0.71).astype(int) == 1) & (
                g['ground_truth'] == 1)).sum() / relevant if relevant > 0 else np.nan
            size_stats.append({'Bin': label, 'Recall': rec, 'N': len(g)})

    size_stats_df = pd.DataFrame(size_stats)
    print(size_stats_df.set_index('Bin').to_string())

    export_worst_guesses_spatial(df, threshold=0.71)

    # --- Plotting ---
    # Set style for cleaner transparent plots
    sns.set_style("white")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    y_pred = (y_score >= 0.71).astype(int)

    # 1. Confusion Matrix
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True,
                fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title(
        f"Confusion Matrix @ 0.71\n(Acc: {accuracy_score(y_true, y_pred):.2%})")

    # 2. Recall by Size
    sns.barplot(data=size_stats_df, x='Bin', y='Recall',
                palette="viridis", ax=axes[1])
    axes[1].axhline(best_tpr, color='red', linestyle='--')
    axes[1].set_title("Recall by Building Size")
    axes[1].set_ylim(0, 1.1)

    # 3. ROC Curve
    axes[2].plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC (AUC = {roc_auc:.3f})')
    axes[2].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    axes[2].scatter(best_fpr, best_tpr, color='red', s=50,
                    label=f'Best Thr: {best_acc_thr:.2f}')
    axes[2].set_title('ROC Curve')
    axes[2].legend(loc="lower right")

    plt.tight_layout()

    # --- Export ---
    os.makedirs(PLOT_EXPORT_PATH, exist_ok=True)

    # Save combined figure
    full_plot_path = os.path.join(PLOT_EXPORT_PATH, "validation_summary.png")
    fig.savefig(full_plot_path, transparent=True, dpi=300, bbox_inches='tight')

    # Optional: Save just the ROC curve as a single image
    # We do this by creating a new small figure for just that axis
    roc_fig, roc_ax = plt.subplots(figsize=(6, 6))
    roc_ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC (AUC = {roc_auc:.3f})')
    roc_ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    roc_ax.scatter(best_fpr, best_tpr, color='red', s=50)
    roc_ax.set_title('ROC Curve')
    roc_ax.legend(loc="lower right")

    roc_only_path = os.path.join(PLOT_EXPORT_PATH, "roc_only_transparent.png")
    roc_fig.savefig(roc_only_path, transparent=True,
                    dpi=300, bbox_inches='tight')

    print(f"✅ Plots saved to: {PLOT_EXPORT_PATH}")
    plt.show()


if __name__ == "__main__":
    if os.path.exists(RESULTS_PATH):
        analyze_and_plot(pd.read_csv(RESULTS_PATH))
    else:
        print(f"File not found: {RESULTS_PATH}")
