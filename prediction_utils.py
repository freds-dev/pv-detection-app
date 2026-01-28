import geopandas as gpd
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.features import rasterize
from joblib import load
import pandas as pd
import numpy as np
import os
import logging
import warnings
import json
import subprocess
from typing import Dict, List, Tuple, Optional
import planetary_computer as pc
from pystac_client import Client
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Global Setup ---
warnings.filterwarnings("ignore")
LOGGER = logging.getLogger("pv-predictor-util")

try:
    HAVE_PC = True
except Exception:
    HAVE_PC = False

# --- Helper Functions (Copied & Adapted from your script) ---


def setup_logging(level=logging.INFO) -> None:
    logging.basicConfig(
        level=level, format="%(levelname)s | %(name)s | %(message)s")
    for name in logging.Logger.manager.loggerDict:
        if name in ["rasterio", "fiona", "fsspec", "urllib3", "botocore", "s3fs", "geopandas"]:
            logging.getLogger(name).setLevel(logging.ERROR)


def sign_asset(href: str) -> str:
    return pc.sign(href) if HAVE_PC else href


def run_r_script(script_path, session_dir, run_aoa=False):
    aoa_flag = "TRUE" if run_aoa else "FALSE"
    try:
        # Note the three arguments after Rscript
        subprocess.run(['Rscript', script_path, session_dir, aoa_flag], check=True)
        return True
    except:
        return False


def save_geotiff_as_colored_png(
    tiff_path: str, png_path: str, cmap_name: str, title: str
) -> Tuple[Optional[float], Optional[float]]:
    if not os.path.exists(tiff_path):
        LOGGER.warning(f"{title} TIFF not found, skipping PNG creation.")
        return None, None
    LOGGER.info(
        f"Converting {title} TIFF to PNG using colormap '{cmap_name}'...")
    with rasterio.open(tiff_path) as src:
        masked_data = src.read(1, masked=True)

    valid_data = masked_data.compressed()
    if valid_data.size == 0:
        LOGGER.warning(f"No valid data in {title} TIFF, skipping.")
        return None, None

    vmin, vmax = np.percentile(valid_data, [2, 98])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm(masked_data), bytes=True)
    rgba[masked_data.mask] = (0, 0, 0, 0)
    Image.fromarray(rgba).save(png_path)
    return vmin, vmax


def create_building_mask(
    gdf_buildings: gpd.GeoDataFrame,
    transform: rasterio.Affine,
    shape: Tuple[int, int],
    crs: rasterio.crs.CRS,
) -> np.ndarray:
    """Rasterizes building footprints to a mask."""
    LOGGER.info("Rasterizing building footprints to create a mask...")
    if gdf_buildings.empty:
        return np.zeros(shape, dtype=np.uint8)

    buildings_reprojected = gdf_buildings.to_crs(crs)

    if buildings_reprojected.empty:
        LOGGER.warning("No buildings found in the target CRS.")
        return np.zeros(shape, dtype=np.uint8)

    mask = rasterize(shapes=[geom for geom in buildings_reprojected.geometry],
                     out_shape=shape, transform=transform, fill=0, default_value=1, dtype=np.uint8)
    LOGGER.info(
        f"Building mask created with {np.sum(mask):,} building pixels.")
    return mask


def stac_search_one(cfg: dict, bbox_4326: Tuple[float, float, float, float]):
    client = Client.open(cfg["stac_api"])
    search = client.search(collections=[cfg["collection"]], bbox=bbox_4326, datetime=cfg["date_range"], query={
                           "eo:cloud_cover": {"lt": int(cfg["cloud_cover"])}}, max_items=100)
    items = list(search.items())
    if not items:
        raise RuntimeError(
            "No Sentinel-2 items match the AOI/date/cloud filters.")
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 1000))
    return items[0]


def read_bands_to_grid(item, band_names: List[str], bbox_target: Tuple[float, float, float, float], target_epsg: int, resolution: Optional[float] = 10.0) -> Tuple[Dict[str, np.ndarray], rasterio.Affine, rasterio.crs.CRS]:
    res = resolution
    dst_crs = rasterio.crs.CRS.from_epsg(target_epsg)
    minx, miny, maxx, maxy = bbox_target
    width, height = int(np.ceil((maxx - minx) / res)
                        ), int(np.ceil((maxy - miny) / res))
    if width <= 0 or height <= 0:
        raise ValueError(f"AOI bounds are invalid: w={width}, h={height}")
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    arrays: Dict[str, np.ndarray] = {}
    for b in band_names:
        asset = item.assets.get(b)
        href = sign_asset(asset.href) if asset else None
        if not href:
            LOGGER.warning(
                f"Band {b} not found in STAC item, filling with NaN.")
            arrays[b] = np.full((height, width), np.nan, dtype=np.float32)
            continue
        with rasterio.open(href) as src:
            with WarpedVRT(src, crs=dst_crs, transform=transform, width=width, height=height, resampling=Resampling.nearest) as vrt:
                arr = vrt.read(1, out_shape=(height, width)).astype(np.float32)
        arr /= 10000.0
        arrays[b] = arr
    return arrays, transform, dst_crs


def assemble_features(model, band_arrays, feature_list=None):
    """
    Stacks band arrays into a 2D feature matrix (N_pixels, N_features).
    """
    # 1. Determine which features to use
    if feature_list is not None:
        feature_names = feature_list
    elif model is not None and hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        # Fallback to just the bands present in the dictionary
        feature_names = sorted(band_arrays.keys())

    # 2. Extract and stack the data
    # We ensure they are flattened in the same order as feature_names
    try:
        stacked_data = np.stack([band_arrays[name].flatten()
                                for name in feature_names], axis=1)
    except KeyError as e:
        raise KeyError(
            f"Missing band in band_arrays: {e}. Check if STAC returned all requested bands.")

    return stacked_data, feature_names


def predict_mask(model, X, shape: Tuple[int, int], threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(X)[:, 1].astype(np.float32)
    pred = (proba >= float(threshold)).astype(np.uint8)
    return proba.reshape(shape), pred.reshape(shape)


def save_png_rgb(bands: Dict[str, np.ndarray], out_png: str) -> None:
    """
    Saves a "natural color" RGB image (B04, B03, B02) with dynamic
    percentile stretching and gamma correction for a more visually
    appealing and "natural" look.
    """
    r, g, b = bands["B04"], bands["B03"], bands["B02"]

    # Fill any NaN (missing data) pixels with 0 (black) before processing.
    # This is safer for percentile calculations and image stacking.
    r = np.nan_to_num(r, nan=0.0)
    g = np.nan_to_num(g, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)

    enhanced_bands = []

    # Apply enhancement to each band *independently*
    for band in [r, g, b]:
        # 1. Find the 2nd and 98th percentiles (dynamic contrast stretch)
        # This clips the darkest 2% and brightest 2% of pixels
        vmin, vmax = np.percentile(band, [2, 98])

        # 2. Normalize the band to 0-1
        if vmin == vmax:
            vmax = vmin + 1  # Avoid division by zero in blank images
        band_norm = (band - vmin) / (vmax - vmin)

        # 3. Clip to 0-1 range
        band_norm = np.clip(band_norm, 0, 1)

        # 4. Apply Gamma Correction (makes darks/mid-tones look more natural)
        # A gamma of 1/2.2 (approx 0.45) is a common standard.
        band_gamma = band_norm ** (1/2.2)

        # 5. Scale to 0-255 for the 8-bit PNG
        band_uint8 = (band_gamma * 255).astype(np.uint8)
        enhanced_bands.append(band_uint8)

    # 6. Stack the enhanced R, G, B bands into a single 3D array
    rgb_image = np.dstack(enhanced_bands)

    # 7. Save the final image
    Image.fromarray(rgb_image).save(out_png)


def save_png_heatmap(proba: np.ndarray, out_png: str) -> None:
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pv_heat", ["green", "yellow", "red"])
    rgba = cmap(norm(proba))
    rgba[np.isnan(proba), 3] = 0  # Transparent NaN
    rgba[..., 3] = rgba[..., 3] * 0.8  # Apply 80% opacity
    Image.fromarray((rgba * 255).astype(np.uint8)).save(out_png)


def save_footprints_as_png(mask: np.ndarray, out_png: str) -> None:
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[mask == 1] = np.array([255, 255, 0, 150], dtype=np.uint8)  # Yellow
    Image.fromarray(rgba).save(out_png)


def get_aoi_from_building_geom(building_geom, target_epsg: int, buffer_m: float):
    """Creates a buffered AOI from a single building geometry."""
    gdf = gpd.GeoDataFrame([{'geometry': building_geom}], crs=4326)
    gdf_target_crs = gdf.to_crs(epsg=target_epsg)

    # Apply buffer in the target CRS (which should be metric)
    gdf_buffered = gdf_target_crs.buffer(buffer_m)

    # Get bounds for STAC search (in 4326) and prediction (in target_epsg)
    bbox_4326 = tuple(gdf_buffered.to_crs(4326).total_bounds)
    bbox_target = tuple(gdf_buffered.total_bounds)

    return gdf_buffered, bbox_target, bbox_4326
