"""
Sentinel-1 Adaptive Classifier
===============================
Uses scene-specific percentile-based thresholds to classify:
- Glacial Lakes: Lowest VV tail + Highest VV-VH tail
- Glaciers: Low VV and VH (but not extreme)
- Non-Glacier: Everything else

Author: Antigravity
Date: 2026-02-02
"""

import os
import glob
import numpy as np
import rasterio
from scipy.ndimage import median_filter
from skimage.morphology import disk, closing
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_S1_DIR = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1"
OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_adaptive_classification"

# Percentile thresholds (adaptive per scene)
LAKE_VV_PERCENTILE = 10      # Lowest 10% VV
LAKE_RATIO_PERCENTILE = 90   # Highest 10% VV-VH ratio
GLACIER_VV_PERCENTILE = 30   # Low-moderate VV (10-30%)
GLACIER_VH_PERCENTILE = 30   # Low-moderate VH (10-30%)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def apply_speckle_filter(data, size=5):
    """Apply Median filter for speckle reduction."""
    return median_filter(data, size=size)

def classify_s1_image_adaptive(tiff_path, output_dir):
    filename = os.path.basename(tiff_path)
    date_str = filename.replace('.tif', '').replace('S1_', '')
    print(f"\nProcessing {date_str}...")
    
    # 1. Load S1 Data
    with rasterio.open(tiff_path) as src:
        vv_raw = src.read(1)
        vh_raw = src.read(2)
        profile = src.profile
        
    # 2. Preprocessing (Data already in dB, apply Speckle Filter)
    vv_filt = apply_speckle_filter(vv_raw, size=5)
    vh_filt = apply_speckle_filter(vh_raw, size=5)
    
    # Calculate VV-VH ratio
    ratio_db = vv_filt - vh_filt
    
    # 3. Compute Scene-Specific Adaptive Thresholds
    # Filter out invalid/nodata pixels (< -50 dB)
    valid_mask = (vv_filt > -50) & (vh_filt > -50)
    
    vv_valid = vv_filt[valid_mask]
    vh_valid = vh_filt[valid_mask]
    ratio_valid = ratio_db[valid_mask]
    
    # Compute percentile thresholds
    vv_lake_thresh = np.percentile(vv_valid, LAKE_VV_PERCENTILE)
    ratio_lake_thresh = np.percentile(ratio_valid, LAKE_RATIO_PERCENTILE)
    
    vv_glacier_low = np.percentile(vv_valid, LAKE_VV_PERCENTILE)
    vv_glacier_high = np.percentile(vv_valid, GLACIER_VV_PERCENTILE)
    vh_glacier_thresh = np.percentile(vh_valid, GLACIER_VH_PERCENTILE)
    
    print(f"  [Adaptive Thresholds]")
    print(f"    Lake: VV < {vv_lake_thresh:.1f} dB, Ratio > {ratio_lake_thresh:.1f} dB")
    print(f"    Glacier: {vv_glacier_low:.1f} < VV < {vv_glacier_high:.1f} dB, VH < {vh_glacier_thresh:.1f} dB")
    
    # 4. Classification Logic (3 Classes)
    # 0 = Non-Glacier (Black)
    # 1 = Glacial Lake (Blue)
    # 2 = Glacier (White)
    
    classified = np.zeros_like(vv_filt, dtype=np.uint8)
    
    # Glacial Lake: Lowest VV tail AND Highest Ratio tail
    lake_mask = (
        (vv_filt < vv_lake_thresh) & 
        (ratio_db > ratio_lake_thresh) &
        valid_mask
    )
    
    # Glacier: Low-moderate VV and low VH (but not in lake extremes)
    glacier_mask = (
        (vv_filt >= vv_glacier_low) &
        (vv_filt < vv_glacier_high) &
        (vh_filt < vh_glacier_thresh) &
        (~lake_mask) &  # Exclude lakes
        valid_mask
    )
    
    # Apply classifications
    classified[lake_mask] = 1
    classified[glacier_mask] = 2
    
    # 5. Post-processing: Majority filter on lakes
    lake_only = (classified == 1)
    lake_cleaned = closing(lake_only, disk(2))
    
    # Update classification with cleaned lakes
    classified[lake_cleaned] = 1
    
    # Statistics
    total_valid = np.sum(valid_mask)
    lake_pct = (np.sum(classified == 1) / total_valid) * 100
    glacier_pct = (np.sum(classified == 2) / total_valid) * 100
    non_glacier_pct = (np.sum(classified == 0) / total_valid) * 100
    
    print(f"  [Classification Stats]")
    print(f"    Lake: {lake_pct:.2f}%")
    print(f"    Glacier: {glacier_pct:.2f}%")
    print(f"    Non-Glacier: {non_glacier_pct:.2f}%")
    
    # 6. Save GeoTIFF
    os.makedirs(output_dir, exist_ok=True)
    out_name = f"Classified_{date_str}.tif"
    out_path = os.path.join(output_dir, out_name)
    
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(classified, 1)
        
    # 7. Visualization
    viz_path = str(out_path).replace('.tif', '.png')
    visualize_classification(classified, viz_path, date_str)
    
    return out_path

def visualize_classification(class_map, out_path, title):
    """
    Create a 3-color PNG.
    0: Black (Non-Glacier)
    1: Blue (Glacial Lake)
    2: White (Glacier)
    """
    colors = ['black', 'blue', 'white'] 
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(class_map, cmap=cmap, norm=norm, interpolation='nearest')
    plt.title(f"Adaptive Classification: {title}\n(Black: Non-Glacier, Blue: Lake, White: Glacier)", 
              fontsize=12, fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', label='Non-Glacier'),
        Patch(facecolor='blue', label='Glacial Lake'),
        Patch(facecolor='white', edgecolor='gray', label='Glacier')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    plt.axis('off')
    
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    if not os.path.exists(INPUT_S1_DIR):
        print(f"Error: S1 Directory not found: {INPUT_S1_DIR}")
        return

    tiff_files = sorted(glob.glob(os.path.join(INPUT_S1_DIR, "S1_*.tif")))
    print(f"Found {len(tiff_files)} Sentinel-1 images.")
    print("=" * 70)
    
    for tiff in tiff_files:
        try:
            classify_s1_image_adaptive(tiff, OUTPUT_DIR)
        except Exception as e:
            print(f"Failed to process {tiff}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("ADAPTIVE CLASSIFICATION COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
