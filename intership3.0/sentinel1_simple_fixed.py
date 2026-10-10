"""
Simple Fixed-Threshold Sentinel-1 Classifier
==============================================
Uses physically-based FIXED thresholds rather than adaptive percentiles.
Based on known SAR backscatter properties of water, ice, and terrain.

Author: Antigravity
Date: 2026-02-02
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import median_filter
from skimage.morphology import disk, closing, remove_small_objects
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration - FIXED THRESHOLDS
# -----------------------------------------------------------------------------
INPUT_S1_DIR = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1"
DEM_PATH = r"f:\NIDM-SRM Drone Project Docs\GLOF project\Data\GLOFDEM.tiff"
OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_simple_fixed"

# FIXED THRESHOLDS based on SAR physics
# Lakes: Very low backscatter + High cross-pol suppression
LAKE_VV_MAX = -18.0        # Water is very dark
LAKE_RATIO_MIN = 10.0      # Water: VV-VH ratio is HIGH (specular)
LAKE_SLOPE_MAX = 8.0       # Lakes must be flat

# Glaciers: Moderate backscatter + Volume scattering
GLACIER_VV_MIN = -18.0
GLACIER_VV_MAX = -8.0
GLACIER_VH_MAX = -18.0     # Ice has low volume scattering
GLACIER_RATIO_MAX = 8.0    # Glaciers have LOWER ratio than water

MIN_LAKE_SIZE = 20         # Minimum 20 pixels for lakes
MIN_GLACIER_SIZE = 50      # Minimum 50 pixels for glaciers

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def apply_speckle_filter(data, size=5):
    """Apply Median filter for speckle reduction."""
    return median_filter(data, size=size)

def calculate_slope(dem_array, transform):
    """Calculate slope in degrees from DEM."""
    res_x = abs(transform[0])
    res_y = abs(transform[4])
    dy, dx = np.gradient(dem_array)
    dy = dy / res_y
    dx = dx / res_x
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad)

def align_dem_to_s1(s1_profile, dem_path):
    """Reproject DEM to match Sentinel-1 grid."""
    with rasterio.open(dem_path) as src_dem:
        dem_reprojected = np.zeros((s1_profile['height'], s1_profile['width']), dtype=np.float32)
        reproject(
            source=rasterio.band(src_dem, 1),
            destination=dem_reprojected,
            src_transform=src_dem.transform,
            src_crs=src_dem.crs,
            dst_transform=s1_profile['transform'],
            dst_crs=s1_profile['crs'],
            resampling=Resampling.bilinear
        )
    return calculate_slope(dem_reprojected, s1_profile['transform'])

def classify_simple_fixed(tiff_path, dem_path, output_dir):
    """
    Simple classification using FIXED thresholds.
    """
    filename = os.path.basename(tiff_path)
    date_str = filename.replace('.tif', '').replace('S1_', '')
    
    print(f"\n{'='*70}")
    print(f"Processing: {date_str}")
    print(f"{'='*70}")
    
    # 1. Load S1 Data
    with rasterio.open(tiff_path) as src:
        vv_raw = src.read(1)
        vh_raw = src.read(2)
        profile = src.profile
    
    # 2. Preprocessing
    vv = apply_speckle_filter(vv_raw, size=5)
    vh = apply_speckle_filter(vh_raw, size=5)
    ratio = vv - vh
    
    # 3. Load DEM slope
    try:
        slope = align_dem_to_s1(profile, dem_path)
        has_dem = True
        print("  [OK] DEM loaded")
    except Exception as e:
        print(f"  [!] No DEM: {e}")
        slope = np.zeros_like(vv)
        has_dem = False
    
    # Valid data
    valid = (vv > -50) & (vh > -50)
    
    print(f"\n  [Fixed Thresholds]")
    print(f"    Lake: VV<{LAKE_VV_MAX}, Ratio>{LAKE_RATIO_MIN}, Slope<{LAKE_SLOPE_MAX}")
    print(f"    Glacier: {GLACIER_VV_MIN}<VV<{GLACIER_VV_MAX}, VH<{GLACIER_VH_MAX}, Ratio<{GLACIER_RATIO_MAX}")
    
    # 4. CLASSIFICATION - Lakes FIRST
    classified = np.zeros_like(vv, dtype=np.uint8)
    
    # LAKES: Very strict criteria
    lake_mask = (
        (vv < LAKE_VV_MAX) &         # Very dark
        (ratio > LAKE_RATIO_MIN) &   # High ratio = water
        valid
    )
    
    if has_dem:
        lake_mask = lake_mask & (slope < LAKE_SLOPE_MAX)
    
    # Clean lakes
    lake_mask = remove_small_objects(lake_mask.astype(bool), min_size=MIN_LAKE_SIZE)
    lake_mask = closing(lake_mask, disk(2))
    
    classified[lake_mask] = 1
    
    # GLACIERS: Moderate backscatter, exclude lakes
    glacier_mask = (
        (vv >= GLACIER_VV_MIN) &
        (vv <= GLACIER_VV_MAX) &
        (vh < GLACIER_VH_MAX) &
        (ratio <= GLACIER_RATIO_MAX) &  # CRITICAL: Lower ratio than lakes
        (~lake_mask) &                   # Exclude lakes
        valid
    )
    
    # Clean glaciers
    glacier_mask = remove_small_objects(glacier_mask.astype(bool), min_size=MIN_GLACIER_SIZE)
    
    classified[glacier_mask] = 2
    
    # Statistics
    total = np.sum(valid)
    lake_pct = (np.sum(classified == 1) / total) * 100 if total > 0 else 0
    glacier_pct = (np.sum(classified == 2) / total) * 100 if total > 0 else 0
    non_pct = 100 - lake_pct - glacier_pct
    
    print(f"\n  [Results]")
    print(f"    Lake:        {lake_pct:6.2f}%")
    print(f"    Glacier:     {glacier_pct:6.2f}%")
    print(f"    Non-Glacier: {non_pct:6.2f}%")
    
    # 5. Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    out_tif = os.path.join(output_dir, f"Classified_{date_str}.tif")
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(classified, 1)
    
    out_png = os.path.join(output_dir, f"Classified_{date_str}.png")
    visualize_simple(classified, out_png, date_str, lake_pct, glacier_pct, non_pct)
    
    print(f"  [OK] Saved outputs")
    
    return out_tif

def visualize_simple(class_map, out_path, title, lake_pct, glacier_pct, non_pct):
    """Simple 3-class visualization."""
    colors = ['#1a1a1a', '#0099ff', '#ffffff']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(class_map, cmap=cmap, norm=norm, interpolation='nearest')
    
    ax.set_title(
        f"Sentinel-1 Classification: {title}\n"
        f"Fixed-Threshold Method\n"
        f"Lake: {lake_pct:.2f}% | Glacier: {glacier_pct:.2f}% | Non-Glacier: {non_pct:.2f}%",
        fontsize=11, fontweight='bold', pad=15
    )
    
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor='#1a1a1a', label='Non-Glacier'),
        Patch(facecolor='#0099ff', label='Glacial Lake'),
        Patch(facecolor='#ffffff', edgecolor='black', linewidth=0.5, label='Glacier')
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=10, framealpha=0.9)
    ax.axis('off')
    
    # Add threshold info
    info = (f"Thresholds:\n"
            f"Lake: VV<{LAKE_VV_MAX}dB, Ratio>{LAKE_RATIO_MIN}dB\n"
            f"Glacier: {GLACIER_VV_MIN}<VV<{GLACIER_VV_MAX}dB, VH<{GLACIER_VH_MAX}dB")
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=7,
            verticalalignment='bottom', bbox=dict(boxstyle='round', 
            facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    if not os.path.exists(INPUT_S1_DIR):
        print(f"[ERROR] Input not found: {INPUT_S1_DIR}")
        return
    
    tiff_files = sorted(glob.glob(os.path.join(INPUT_S1_DIR, "S1_*.tif")))
    
    print("\n" + "="*70)
    print("SIMPLE FIXED-THRESHOLD S1 CLASSIFIER")
    print("="*70)
    print(f"Images: {len(tiff_files)}")
    print("\nUsing FIXED thresholds based on SAR physics:")
    print(f"  - Lakes: VV < {LAKE_VV_MAX} dB AND Ratio > {LAKE_RATIO_MIN} dB")
    print(f"  - Glaciers: {GLACIER_VV_MIN} < VV < {GLACIER_VV_MAX} dB")
    print(f"  - Key: Ratio (VV-VH) separates lakes from glaciers/shadows")
    print("="*70)
    
    results = []
    for i, tiff in enumerate(tiff_files, 1):
        print(f"\n[{i}/{len(tiff_files)}]", end=" ")
        try:
            result = classify_simple_fixed(tiff, DEM_PATH, OUTPUT_DIR)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"[DONE] {len(results)}/{len(tiff_files)} images processed")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
