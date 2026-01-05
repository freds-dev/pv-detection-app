import geopandas as gpd
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.features import rasterize
import pandas as pd
import numpy as np
import os
import logging
import json
import subprocess
import re
from typing import Dict, List, Tuple, Optional
import planetary_computer as pc
from pystac_client import Client
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

LOGGER = logging.getLogger("pv-predictor-util")


def setup_logging(level=logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def clean_str(s: str) -> str:
    """Removes hidden characters to ensure parity between Python and R."""
    return re.sub(r'[^a-zA-Z0-9_]', '', str(s))


def run_r_script(script_path: str, working_dir: str):
    # Absolute root for path mapping in R
    project_root = os.path.abspath(os.getcwd())
    try:
        env = os.environ.copy()
        env["PROJECT_ROOT"] = project_root
        result = subprocess.run(
            ["Rscript", script_path],
            check=True, capture_output=True, text=True, cwd=working_dir, env=env
        )
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"R script failed: {e.stderr}")
        return False


def read_bands_to_grid(item, band_names, bbox_target, target_epsg, resolution=10.0):
    dst_crs = rasterio.crs.CRS.from_epsg(target_epsg)
    minx, miny, maxx, maxy = bbox_target
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    arrays = {}
    for b in band_names:
        asset = item.assets.get(b)
        href = pc.sign(asset.href) if asset else None
        if not href:
            arrays[b] = np.full((height, width), np.nan, dtype=np.float32)
            continue
        with rasterio.open(href) as src:
            # BILINEAR is required to upscale B11/B12 (20m) to 10m grid
            with WarpedVRT(src, crs=dst_crs, transform=transform, width=width, height=height,
                           resampling=Resampling.bilinear) as vrt:
                arrays[b] = vrt.read(1).astype(np.float32) / 10000.0
    return arrays, transform, dst_crs


def assemble_features(model, band_arrays: Dict[str, np.ndarray], item_datetime) -> Tuple[pd.DataFrame, List[str]]:
    H, W = next(iter(band_arrays.values())).shape
    raw_features = list(getattr(model, "feature_names_in_", []))
    clean_features = [clean_str(f) for f in raw_features]

    month = item_datetime.month
    if month in [12, 1, 2]:   active = "winter"
    elif month in [3, 4, 5]:  active = "spring"
    elif month in [6, 7, 8]:  active = "summer"
    else:                     active = "autumn"

    stack = []
    for clean_f in clean_features:
        if clean_f in band_arrays:
            stack.append(band_arrays[clean_f].ravel())
        elif "season" in clean_f.lower():
            val = 1.0 if active in clean_f.lower() else 0.0
            stack.append(np.full(H * W, val, dtype=np.float32))
        elif clean_f in ("x", "y"):
            ys, xs = np.indices((H, W))
            stack.append((xs if clean_f == "x" else ys).ravel().astype(np.float32))
        else:
            stack.append(np.zeros(H * W, dtype=np.float32))

    # CRITICAL FIX: Return a DataFrame with raw_features as columns
    X_df = pd.DataFrame(np.vstack(stack).T, columns=raw_features)
    X_df = X_df.fillna(0.0)
    
    return X_df, raw_features

def predict_mask(model, X_df, shape: Tuple[int, int], threshold: float):
    # scikit-learn is happy now because X_df has the correct feature names
    proba = model.predict_proba(X_df)[:, 1].astype(np.float32)
    pred = (proba >= float(threshold)).astype(np.uint8)
    return proba.reshape(shape), pred.reshape(shape)


def save_png_heatmap(proba, out_png):
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap("RdYlGn_r")
    rgba = cmap(norm(proba))
    rgba[np.isnan(proba), 3] = 0
    Image.fromarray((rgba * 255).astype(np.uint8)).save(out_png)


def save_png_rgb(bands, out_png):
    rgb = []
    for b in ["B04", "B03", "B02"]:
        arr = np.nan_to_num(bands[b])
        vmin, vmax = np.percentile(arr, [2, 98])
        arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0, 1)
        rgb.append((arr**(1/2.2) * 255).astype(np.uint8))
    Image.fromarray(np.dstack(rgb)).save(out_png)


def save_geotiff_as_colored_png(tiff_path, png_path, cmap_name, title):
    with rasterio.open(tiff_path) as src:
        data = src.read(1, masked=True)
    if data.compressed().size == 0:
        return
    vmin, vmax = np.percentile(data.compressed(), [2, 98])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = plt.get_cmap(cmap_name)(norm(data), bytes=True)
    rgba[data.mask] = (0, 0, 0, 0)
    Image.fromarray(rgba).save(png_path)


def create_building_mask(gdf, transform, shape, crs):
    reproj = gdf.to_crs(crs)
    return rasterize(shapes=reproj.geometry, out_shape=shape, transform=transform, fill=0, default_value=1, dtype=np.uint8)


def get_aoi_from_building_geom(geom, epsg, buffer):
    gdf = gpd.GeoDataFrame([{'geometry': geom}], crs=4326).to_crs(epsg)
    buffered = gdf.buffer(buffer)
    return buffered, tuple(buffered.total_bounds), tuple(buffered.to_crs(4326).total_bounds)


def stac_search_one(cfg, bbox):
    client = Client.open(cfg["stac_api"])
    search = client.search(collections=[cfg["collection"]], bbox=bbox, datetime=cfg["date_range"],
                           query={"eo:cloud_cover": {"lt": int(cfg["cloud_cover"])}}, max_items=20)
    items = list(search.items())
    if not items:
        raise RuntimeError("No scenes found")
    items.sort(key=lambda x: x.properties['eo:cloud_cover'])
    return items[0]
