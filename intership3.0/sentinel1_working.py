"""
Sentinel-1 Classifier - RELAXED Thresholds
===========================================
More liberal thresholds to actually detect lakes and glaciers.
Easy to adjust if needed.

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

# =============================================================================
# ADJUST THESE THRESHOLDS TO GET BETTER RESULTS
# =============================================================================
INPUT_S1_DIR = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1"
DEM_PATH = r"f:\NIDM-SRM Drone Project Docs\GLOF project\Data\GLOFDEM.tiff"
OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_working"

# LAKES - Adjusted to be MORE INCLUSIVE
LAKE_VV_MAX = -12.0        # Increased from -18 (catches more lakes)
LAKE_RATIO_MIN = 7.0       # Decreased from 10 (more inclusive)
LAKE_SLOPE_MAX = 12.0      # Increased from 8 (allow slightly sloped areas)

# GLACIERS
GLACIER_VV_MIN = -16.0     
GLACIER_VV_MAX = -8.0
GLACIER_VH_MAX = -16.0
GLACIER_RATIO_MAX = 7.0    # Glaciers have LOWER ratio than lakes

# Minimum sizes (pixels)
MIN_LAKE_SIZE = 10
MIN_GLACIER_SIZE = 30

# =============================================================================

def apply_speckle_filter(data, size=5):
    return median_filter(data, size=size)

def calculate_slope(dem_array, transform):
    res_x = abs(transform[0])
    res_y = abs(transform[4])
    dy, dx = np.gradient(dem_array)
    dy = dy / res_y
    dx = dx / res_x
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad)

def align_dem_to_s1(s1_profile, dem_path):
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

def classify_working(tiff_path, dem_path, output_dir):
    filename = os.path.basename(tiff_path)
    date_str = filename.replace('.tif', '').replace('S1_', '')
    
    print(f"\n{'='*70}")
    print(f"Processing: {date_str}")
    print(f"{'='*70}")
    
    # Load data
    with rasterio.open(tiff_path) as src:
        vv_raw = src.read(1)
        vh_raw = src.read(2)
        profile = src.profile
    
    # Preprocess
    vv = apply_speckle_filter(vv_raw, size=5)
    vh = apply_speckle_filter(vh_raw, size=5)
    ratio = vv - vh
    
    # Load DEM
    try:
        slope = align_dem_to_s1(profile, dem_path)
        has_dem = True
    except Exception as e:
        print(f"  [!] No DEM: {e}")
        slope = np.zeros_like(vv)
        has_dem = False
    
    valid = (vv > -50) & (vh > -50)
    
    # Show data statistics
    vv_stats = np.percentile(vv[valid], [5, 25, 50, 75, 95])
    ratio_stats = np.percentile(ratio[valid], [5, 25, 50, 75, 95])
    
    print(f"\n  [Data Stats]")
    print(f"    VV:    P05={vv_stats[0]:.1f}, P50={vv_stats[2]:.1f}, P95={vv_stats[4]:.1f}")
    print(f"    Ratio: P05={ratio_stats[0]:.1f}, P50={ratio_stats[2]:.1f}, P95={ratio_stats[4]:.1f}")
    
    print(f"\n  [Thresholds]")
    print(f"    Lake:    VV<{LAKE_VV_MAX}, Ratio>{LAKE_RATIO_MIN}, Slope<{LAKE_SLOPE_MAX}")
    print(f"    Glacier: {GLACIER_VV_MIN}<VV<{GLACIER_VV_MAX}, VH<{GLACIER_VH_MAX}, Ratio<{GLACIER_RATIO_MAX}")
    
    # CLASSIFICATION
    classified = np.zeros_like(vv, dtype=np.uint8)
    
    # LAKES FIRST (priority)
    lake_mask = (
        (vv < LAKE_VV_MAX) &
        (ratio > LAKE_RATIO_MIN) &
        valid
    )
    
    if has_dem:
        lake_mask = lake_mask & (slope < LAKE_SLOPE_MAX)
    
    # Clean lakes
    lake_mask = remove_small_objects(lake_mask.astype(bool), min_size=MIN_LAKE_SIZE)
    lake_mask = closing(lake_mask, disk(2))
    
    classified[lake_mask] = 1
    
    # GLACIERS (exclude lakes)
    glacier_mask = (
        (vv >= GLACIER_VV_MIN) &
        (vv <= GLACIER_VV_MAX) &
        (vh < GLACIER_VH_MAX) &
        (ratio <= GLACIER_RATIO_MAX) &
        (~lake_mask) &
        valid
    )
    
    glacier_mask = remove_small_objects(glacier_mask.astype(bool), min_size=MIN_GLACIER_SIZE)
    
    classified[glacier_mask] = 2
    
    # Stats
    total = np.sum(valid)
    lake_pct = (np.sum(classified == 1) / total) * 100
    glacier_pct = (np.sum(classified == 2) / total) * 100
    non_pct = 100 - lake_pct - glacier_pct
    
    print(f"\n  [Results]")
    print(f"    Lake:        {lake_pct:6.2f}%")
    print(f"    Glacier:     {glacier_pct:6.2f}%")
    print(f"    Non-Glacier: {non_pct:6.2f}%")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    
    out_tif = os.path.join(output_dir, f"Classified_{date_str}.tif")
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(classified, 1)
    
    out_png = os.path.join(output_dir, f"Classified_{date_str}.png")
    visualize(classified, out_png, date_str, lake_pct, glacier_pct, non_pct, vv_stats, ratio_stats)
    
    print(f"  [OK] Saved")
    
    return out_tif

def visualize(class_map, out_path, title, lake_pct, glacier_pct, non_pct, vv_stats, ratio_stats):
    colors = ['#1a1a1a', '#00BFFF', '#FFFFFF']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.imshow(class_map, cmap=cmap, norm=norm, interpolation='nearest')
    
    ax.set_title(
        f"Sentinel-1 Classification: {title}\n"
        f"Lake: {lake_pct:.2f}% | Glacier: {glacier_pct:.2f}% | Non-Glacier: {non_pct:.2f}%",
        fontsize=12, fontweight='bold', pad=15
    )
    
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor='#1a1a1a', label=f'Non-Glacier ({non_pct:.1f}%)'),
        Patch(facecolor='#00BFFF', label=f'Glacial Lake ({lake_pct:.1f}%)'),
        Patch(facecolor='#FFFFFF', edgecolor='black', linewidth=0.5, label=f'Glacier ({glacier_pct:.1f}%)')
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=10, framealpha=0.9)
    ax.axis('off')
    
    # Info box
    info = (f"Thresholds:\n"
            f"Lake: VV<{LAKE_VV_MAX}dB, Ratio>{LAKE_RATIO_MIN}dB\n"
            f"Glacier: {GLACIER_VV_MIN}<VV<{GLACIER_VV_MAX}dB\n\n"
            f"Data Range:\n"
            f"VV: {vv_stats[0]:.1f} to {vv_stats[4]:.1f} dB\n"
            f"Ratio: {ratio_stats[0]:.1f} to {ratio_stats[4]:.1f} dB")
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=7.5,
            verticalalignment='bottom', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    if not os.path.exists(INPUT_S1_DIR):
        print(f"[ERROR] Input not found: {INPUT_S1_DIR}")
        return
    
    tiff_files = sorted(glob.glob(os.path.join(INPUT_S1_DIR, "S1_*.tif")))
    
    print("\n" + "="*70)
    print("WORKING SENTINEL-1 CLASSIFIER")
    print("="*70)
    print(f"Images: {len(tiff_files)}")
    print("\n** RELAXED THRESHOLDS - EASIER TO ADJUST **")
    print(f"  Lake:    VV < {LAKE_VV_MAX} dB, Ratio > {LAKE_RATIO_MIN} dB")
    print(f"  Glacier: {GLACIER_VV_MIN} < VV < {GLACIER_VV_MAX} dB, Ratio < {GLACIER_RATIO_MAX} dB")
    print("\nIf results not good, EDIT THE THRESHOLDS at top of script!")
    print("="*70)
    
    results = []
    for i, tiff in enumerate(tiff_files, 1):
        print(f"\n[{i}/{len(tiff_files)}]", end=" ")
        try:
            result = classify_working(tiff, DEM_PATH, OUTPUT_DIR)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"[DONE] {len(results)}/{len(tiff_files)} processed")
    print(f"Output: {OUTPUT_DIR}")
    print("\nCheck the PNG files to see if classification is good!")
    print("If not, ADJUST THE THRESHOLDS at the top of the script.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
