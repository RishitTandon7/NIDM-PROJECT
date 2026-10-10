"""
Robust Sentinel-1 3-Class Classifier
=====================================
Separates Glacier, Glacial Lake, and Non-Glacier using multiple features
to distinguish lakes from shadows (both have similar darkness/blackness).

Key Strategy:
- Lakes vs Shadows: Use VV-VH ratio + terrain slope + texture
- Glaciers: Use combined VV+VH characteristics + slope
- Everything else: Non-Glacier

Author: Antigravity
Date: 2026-02-02
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import median_filter, generic_filter
from skimage.morphology import disk, closing, binary_erosion
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_S1_DIR = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1"
DEM_PATH = r"f:\NIDM-SRM Drone Project Docs\GLOF project\Data\GLOFDEM.tiff"
OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_robust_3class"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def apply_speckle_filter(data, size=5):
    """Apply Median filter for speckle reduction."""
    return median_filter(data, size=size)

def calculate_texture_variance(data, size=7):
    """Calculate local variance as texture measure."""
    def local_var(values):
        return np.var(values)
    return generic_filter(data, local_var, size=size, mode='nearest')

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
    return dem_reprojected, calculate_slope(dem_reprojected, s1_profile['transform'])

def classify_robust_3class(tiff_path, dem_path, output_dir):
    """
    Robust 3-class classification using multiple features.
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
    
    print("  [OK] Loaded VV and VH bands")
    
    # 2. Preprocessing
    vv = apply_speckle_filter(vv_raw, size=5)
    vh = apply_speckle_filter(vh_raw, size=5)
    
    # 3. Feature Extraction
    print("  [*] Extracting features...")
    
    # Feature 1: VV-VH Ratio (KEY for separating water from shadows)
    # Water has HIGH ratio (specular reflection), shadows have LOWER ratio
    ratio_db = vv - vh
    
    # Feature 2: Texture (water is smooth, terrain/shadows are rough)
    vv_texture = calculate_texture_variance(vv, size=7)
    
    # Feature 3: Terrain slope from DEM
    try:
        dem, slope = align_dem_to_s1(profile, dem_path)
        has_dem = True
        print("  [OK] DEM aligned and slope computed")
    except Exception as e:
        print(f"  [!] DEM error: {e}")
        slope = np.zeros_like(vv)
        has_dem = False
    
    # Valid data mask
    valid = (vv > -50) & (vh > -50)
    
    # 4. Compute Adaptive Thresholds
    # ADJUSTED: More inclusive lake detection
    vv_p05 = np.percentile(vv[valid], 5)
    vv_p15 = np.percentile(vv[valid], 15)   # Lakes: increased from P10
    vv_p20 = np.percentile(vv[valid], 20)
    vv_p30 = np.percentile(vv[valid], 30)
    
    vh_p15 = np.percentile(vh[valid], 15)
    vh_p30 = np.percentile(vh[valid], 30)
    
    ratio_p80 = np.percentile(ratio_db[valid], 80)  # Lakes: decreased from P90
    ratio_p85 = np.percentile(ratio_db[valid], 85)
    
    texture_median = np.median(vv_texture[valid])
    texture_p30 = np.percentile(vv_texture[valid], 30)
    
    print(f"\n  [Adaptive Thresholds]")
    print(f"    VV: P05={vv_p05:.1f}, P15={vv_p15:.1f}, P20={vv_p20:.1f}, P30={vv_p30:.1f}")
    print(f"    VH: P15={vh_p15:.1f}, P30={vh_p30:.1f}")
    print(f"    Ratio: P80={ratio_p80:.1f}, P85={ratio_p85:.1f}")
    print(f"    Texture: Median={texture_median:.2f}, P30={texture_p30:.2f}")
    
    # 5. CLASSIFICATION LOGIC
    print("\n  [*] Classifying...")
    
    # Initialize as Non-Glacier (class 0)
    classified = np.zeros_like(vv, dtype=np.uint8)
    
    # -------------------------------------------------------------------------
    # CLASS 1: GLACIAL LAKES (PRIORITY - Detect FIRST)
    # -------------------------------------------------------------------------
    # MORE INCLUSIVE criteria to avoid missing lakes:
    # 1. Low VV (< P15) - RELAXED from P10
    # 2. High VV-VH ratio (> P80) - RELAXED from P90
    # 3. Low texture (smooth surface)
    # 4. Flat terrain if DEM available
    
    # Primary lake detection
    lake_candidates = (
        (vv < vv_p15) &            # Increased threshold to catch more lakes
        (ratio_db > ratio_p80) &   # Decreased threshold to catch more lakes
        (vv_texture < texture_p30) &  # Smooth surfaces
        valid
    )
    
    # Secondary lake detection (backup for very dark lakes with lower ratio)
    lake_candidates_alt = (
        (vv < vv_p05) &            # Very dark
        (ratio_db > ratio_p85) &   # Still relatively high ratio
        valid
    )
    
    # Combine lake detections
    lake_combined = lake_candidates | lake_candidates_alt
    
    # Add slope constraint if DEM is available
    if has_dem:
        lake_mask = lake_combined & (slope < 10.0)  # Increased from 8 to 10
        print(f"    Lake: VV<{vv_p15:.1f}, Ratio>{ratio_p80:.1f}, Slope<10deg")
    else:
        lake_mask = lake_combined
        print(f"    Lake: VV<{vv_p15:.1f}, Ratio>{ratio_p80:.1f}")
    
    # Clean lakes: remove tiny noise, fill holes
    from scipy.ndimage import binary_opening
    lake_mask = binary_opening(lake_mask, disk(1))  # Remove single-pixel noise
    lake_mask = closing(lake_mask, disk(3))          # Fill holes
    
    # ASSIGN LAKE CLASS
    classified[lake_mask] = 1
    
    # -------------------------------------------------------------------------
    # CLASS 2: GLACIERS
    # -------------------------------------------------------------------------
    # Glaciers have:
    # 1. Moderate VV (P15 to P30)
    # 2. Low VH (< P30)
    # 3. EXPLICITLY EXCLUDE LAKES (critical!)
    
    glacier_mask = (
        (vv >= vv_p15) &           # Above lake threshold
        (vv <= vv_p30) &
        (vh < vh_p30) &
        (~lake_mask) &             # CRITICAL: Exclude detected lakes
        (ratio_db <= ratio_p80) &  # Lower ratio than lakes
        valid
    )
    
    # ASSIGN GLACIER CLASS
    classified[glacier_mask] = 2
    
    print(f"    Glacier: {vv_p15:.1f}<VV<{vv_p30:.1f}, VH<{vh_p30:.1f}, Ratio<{ratio_p80:.1f}")
    
    # -------------------------------------------------------------------------
    # CLASS 0: NON-GLACIER (default for everything else)
    # -------------------------------------------------------------------------
    # Includes: land, vegetation, shadows, steep terrain
    
    # 6. Statistics
    total = np.sum(valid)
    lake_pct = (np.sum(classified == 1) / total) * 100
    glacier_pct = (np.sum(classified == 2) / total) * 100
    non_glacier_pct = (np.sum(classified == 0) / total) * 100
    
    print(f"\n  [Results]")
    print(f"    Glacial Lake:  {lake_pct:6.2f}%")
    print(f"    Glacier:       {glacier_pct:6.2f}%")
    print(f"    Non-Glacier:   {non_glacier_pct:6.2f}%")
    
    # 7. Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    # Save GeoTIFF
    out_tif = os.path.join(output_dir, f"Classified_{date_str}.tif")
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(classified, 1)
    
    # Save visualization
    out_png = os.path.join(output_dir, f"Classified_{date_str}.png")
    visualize_3class(classified, out_png, date_str, lake_pct, glacier_pct, non_glacier_pct)
    
    print(f"\n  [OK] Saved: {os.path.basename(out_tif)}")
    print(f"  [OK] Saved: {os.path.basename(out_png)}")
    
    return out_tif

def visualize_3class(class_map, out_path, title, lake_pct, glacier_pct, non_glacier_pct):
    """
    Visualize 3-class classification.
    0=Black (Non-Glacier), 1=Blue (Lake), 2=White (Glacier)
    """
    colors = ['#2F2F2F', '#0066FF', '#F0F0F0']  # Dark gray, Blue, Light gray
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(class_map, cmap=cmap, norm=norm, interpolation='nearest')
    
    ax.set_title(
        f"Sentinel-1 Classification: {title}\n"
        f"Lake: {lake_pct:.2f}% | Glacier: {glacier_pct:.2f}% | Non-Glacier: {non_glacier_pct:.2f}%",
        fontsize=13, fontweight='bold', pad=20
    )
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2F2F2F', label=f'Non-Glacier ({non_glacier_pct:.1f}%)'),
        Patch(facecolor='#0066FF', label=f'Glacial Lake ({lake_pct:.1f}%)'),
        Patch(facecolor='#F0F0F0', edgecolor='black', linewidth=0.5, label=f'Glacier ({glacier_pct:.1f}%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
              framealpha=0.95, edgecolor='black', fancybox=True)
    ax.axis('off')
    
    # Add methodology note
    note = ("Method: Multi-feature separation\n"
            "• Lakes: Low VV + High Ratio + Low Texture + Flat\n"
            "• Glaciers: Moderate VV + Low VH\n"
            "• Shadows suppressed using slope & ratio")
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    if not os.path.exists(INPUT_S1_DIR):
        print(f"[ERROR] Input directory not found: {INPUT_S1_DIR}")
        return
    
    tiff_files = sorted(glob.glob(os.path.join(INPUT_S1_DIR, "S1_*.tif")))
    
    print("\n" + "="*70)
    print("ROBUST 3-CLASS SENTINEL-1 CLASSIFIER")
    print("="*70)
    print(f"Input: {len(tiff_files)} Sentinel-1 images")
    print("""\nStrategy for Lake vs Shadow separation:
  - VV-VH Ratio: Water has HIGH ratio (specular)
  - Texture: Water is SMOOTH
  - Slope: Lakes on FLAT terrain, shadows on STEEP
  - Combined approach ensures robust discrimination""")
    print("="*70)
    
    results = []
    for i, tiff in enumerate(tiff_files, 1):
        print(f"\n[{i}/{len(tiff_files)}]", end=" ")
        try:
            result = classify_robust_3class(tiff, DEM_PATH, OUTPUT_DIR)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("[SUCCESS] CLASSIFICATION COMPLETE!")
    print("="*70)
    print(f"  Success: {len(results)}/{len(tiff_files)} images")
    print(f"  Output: {OUTPUT_DIR}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
