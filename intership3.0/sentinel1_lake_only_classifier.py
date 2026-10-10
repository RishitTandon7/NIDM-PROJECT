"""
Sentinel-1 Lake-Only Classifier
================================
"Due to severe radar shadow and terrain-induced backscatter overlap, 
Sentinel-1 intensity data were used only for robust glacial lake extraction, 
while glacier extents were derived from optical or inventory-based masks."

This classifier focuses exclusively on detecting glacial lakes using:
- Adaptive VV thresholds (lowest percentile)
- High VV-VH ratio (specular reflection characteristic of water)
- Slope-based shadow suppression

Author: Antigravity
Date: 2026-02-02
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import median_filter
from skimage.morphology import disk, closing, remove_small_objects as remove_small
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_S1_DIR = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1"
DEM_PATH = r"f:\NIDM-SRM Drone Project Docs\GLOF project\Data\GLOFDEM.tiff"
OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_lakes_only"

# Adaptive thresholds
LAKE_VV_PERCENTILE = 8       # Lowest 8% VV (very low backscatter)
LAKE_RATIO_PERCENTILE = 92   # Highest 8% VV-VH ratio (specular reflection)
SLOPE_THRESHOLD = 10.0       # Exclude slopes > 10° to suppress shadows
MIN_LAKE_SIZE = 10           # Minimum lake size in pixels

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
    """Reproject DEM to match Sentinel-1 grid and calculate slope."""
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

def extract_lakes_only(tiff_path, dem_path, output_dir):
    """
    Extract glacial lakes only from Sentinel-1 data.
    Returns binary classification: 0 = Non-Lake, 1 = Glacial Lake
    """
    filename = os.path.basename(tiff_path)
    date_str = filename.replace('.tif', '').replace('S1_', '')
    print(f"\n{'='*70}")
    print(f"Processing {date_str}")
    print(f"{'='*70}")
    
    # 1. Load S1 Data
    with rasterio.open(tiff_path) as src:
        vv_raw = src.read(1)
        vh_raw = src.read(2)
        profile = src.profile
        
    # 2. Preprocessing
    vv_filt = apply_speckle_filter(vv_raw, size=5)
    vh_filt = apply_speckle_filter(vh_raw, size=5)
    ratio_db = vv_filt - vh_filt
    
    # 3. Load and align DEM for slope-based shadow masking
    try:
        slope_map = align_dem_to_s1(profile, dem_path)
        flat_terrain = slope_map < SLOPE_THRESHOLD
        print(f"  ✓ DEM aligned, slope threshold: {SLOPE_THRESHOLD}°")
    except Exception as e:
        print(f"  ⚠ DEM error: {e}, proceeding without slope mask")
        flat_terrain = np.ones_like(vv_filt, dtype=bool)
    
    # 4. Compute adaptive thresholds from valid pixels
    valid_mask = (vv_filt > -50) & (vh_filt > -50) & flat_terrain
    
    vv_valid = vv_filt[valid_mask]
    ratio_valid = ratio_db[valid_mask]
    
    vv_lake_thresh = np.percentile(vv_valid, LAKE_VV_PERCENTILE)
    ratio_lake_thresh = np.percentile(ratio_valid, LAKE_RATIO_PERCENTILE)
    
    print(f"\n  [Adaptive Thresholds]")
    print(f"    VV (lowest {LAKE_VV_PERCENTILE}%): < {vv_lake_thresh:.2f} dB")
    print(f"    VV-VH Ratio (highest {100-LAKE_RATIO_PERCENTILE}%): > {ratio_lake_thresh:.2f} dB")
    print(f"    Flat terrain: Slope < {SLOPE_THRESHOLD}°")
    
    # 5. Lake Detection
    # Criteria:
    # - Very low VV backscatter (smooth surface)
    # - High VV-VH ratio (specular reflection)
    # - Flat terrain (suppress radar shadows)
    lake_mask = (
        (vv_filt < vv_lake_thresh) &
        (ratio_db > ratio_lake_thresh) &
        flat_terrain &
        valid_mask
    )
    
    # 6. Post-processing
    # Remove small isolated pixels
    lake_mask_cleaned = remove_small(lake_mask, min_size=MIN_LAKE_SIZE)
    
    # Morphological closing to fill small holes
    lake_mask_final = closing(lake_mask_cleaned, disk(2))
    
    # Create binary classification
    classified = np.zeros_like(vv_filt, dtype=np.uint8)
    classified[lake_mask_final] = 1
    
    # Statistics
    total_valid = np.sum(valid_mask)
    lake_pixels = np.sum(classified == 1)
    lake_pct = (lake_pixels / total_valid) * 100 if total_valid > 0 else 0
    
    print(f"\n  [Detection Results]")
    print(f"    Glacial Lake pixels: {lake_pixels:,} ({lake_pct:.3f}%)")
    print(f"    Non-Lake pixels: {total_valid - lake_pixels:,} ({100-lake_pct:.3f}%)")
    
    # 7. Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    # Save GeoTIFF
    out_tif = os.path.join(output_dir, f"Lake_{date_str}.tif")
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(classified, 1)
    
    # Save visualization
    out_png = os.path.join(output_dir, f"Lake_{date_str}.png")
    visualize_lakes(classified, out_png, date_str, lake_pct)
    
    print(f"\n  ✓ Saved: {os.path.basename(out_tif)}")
    print(f"  ✓ Saved: {os.path.basename(out_png)}")
    
    return out_tif

def visualize_lakes(lake_map, out_path, title, lake_pct):
    """
    Create visualization showing only lakes (blue) on black background.
    """
    colors = ['black', '#00BFFF']  # Black for non-lake, Deep Sky Blue for lake
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(lake_map, cmap=cmap, norm=norm, interpolation='nearest')
    
    ax.set_title(
        f"Glacial Lake Detection: {title}\n"
        f"Lake Coverage: {lake_pct:.3f}%\n"
        f"(Sentinel-1 used for lake extraction only)",
        fontsize=11, fontweight='bold', pad=15
    )
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', label='Non-Lake (Terrain/Glacier/Shadow)'),
        Patch(facecolor='#00BFFF', label='Glacial Lake')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
              framealpha=0.9, edgecolor='gray')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    if not os.path.exists(INPUT_S1_DIR):
        print(f"❌ Error: S1 Directory not found: {INPUT_S1_DIR}")
        return
    
    if not os.path.exists(DEM_PATH):
        print(f"⚠ Warning: DEM not found: {DEM_PATH}")
        print("  Proceeding without slope-based shadow suppression")

    tiff_files = sorted(glob.glob(os.path.join(INPUT_S1_DIR, "S1_*.tif")))
    
    print("\n" + "="*70)
    print("SENTINEL-1 GLACIAL LAKE EXTRACTION")
    print("="*70)
    print(f"Found {len(tiff_files)} Sentinel-1 images")
    print("\nMethodology:")
    print("  • Sentinel-1 used ONLY for lake detection")
    print("  • Glacier mapping requires optical/inventory data")
    print("  • Reason: Severe radar shadow & terrain backscatter overlap")
    print("="*70)
    
    lake_files = []
    for tiff in tiff_files:
        try:
            lake_file = extract_lakes_only(tiff, DEM_PATH, OUTPUT_DIR)
            lake_files.append(lake_file)
        except Exception as e:
            print(f"\n❌ Failed to process {tiff}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✓ LAKE EXTRACTION COMPLETE!")
    print("="*70)
    print(f"  Processed: {len(lake_files)}/{len(tiff_files)} images")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"\n  Note: Glacier extents should be derived from:")
    print(f"    - Optical imagery (Sentinel-2, Landsat)")
    print(f"    - Existing glacier inventories (RGI, etc.)")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
