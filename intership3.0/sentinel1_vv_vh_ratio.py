"""
Sentinel-1 VV/VH Ratio-Based Classification
============================================
Uses VV/VH polarization ratio for lake detection:
- Water: High VV/VH ratio (specular reflection keeps polarization)
- Rough terrain: Lower ratio (depolarization from scattering)
- Shadows: Different ratio profile than actual water

Author: Antigravity
Date: 2026-01-30
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
from PIL import Image
from scipy.ndimage import uniform_filter, median_filter
from scipy.ndimage import binary_closing
from skimage import morphology
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def load_sentinel1(tiff_path):
    """Load Sentinel-1 data"""
    with rasterio.open(tiff_path) as src:
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)
        profile = src.profile.copy()
    return vv, vh, profile


def denoise(data, size=5):
    """Simple denoising"""
    data = median_filter(data, size=3)
    # Lee filter
    local_mean = uniform_filter(data.astype(np.float64), size)
    local_sqr_mean = uniform_filter(data.astype(np.float64)**2, size)
    local_var = np.maximum(local_sqr_mean - local_mean**2, 0)
    overall_var = np.var(data)
    w = local_var / (local_var + overall_var + 1e-10)
    return (local_mean + w * (data - local_mean)).astype(np.float32)


def compute_vv_vh_ratio(vv, vh):
    """
    Compute VV/VH ratio (in dB, this is VV - VH).
    
    Since data is in dB:
    ratio_dB = VV_dB - VH_dB = 10*log10(VV_linear/VH_linear)
    
    Typical values:
    - Water: ratio > 7-8 dB (VV much stronger than VH)
    - Glacier: ratio 5-10 dB
    - Rough terrain: ratio 3-6 dB
    - Very rough: ratio < 4 dB
    """
    ratio = vv - vh  # In dB, ratio = VV - VH
    return ratio


def analyze_vv_vh_ratio(vv, vh, output_dir, date_str):
    """
    Analyze VV/VH ratio statistics and create visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ratio = compute_vv_vh_ratio(vv, vh)
    
    # Statistics
    print(f"\n    VV/VH Ratio Statistics:")
    print(f"    Min:  {np.nanmin(ratio):.2f} dB")
    print(f"    Max:  {np.nanmax(ratio):.2f} dB")
    print(f"    Mean: {np.nanmean(ratio):.2f} dB")
    print(f"    Std:  {np.nanstd(ratio):.2f} dB")
    print(f"    Percentiles: 5%={np.percentile(ratio, 5):.1f}, 50%={np.percentile(ratio, 50):.1f}, 95%={np.percentile(ratio, 95):.1f}")
    
    # Create ratio visualization
    h, w = ratio.shape
    
    # Normalize ratio for visualization
    r_min, r_max = np.percentile(ratio, [2, 98])
    ratio_norm = np.clip((ratio - r_min) / (r_max - r_min + 1e-10), 0, 1)
    ratio_8bit = (ratio_norm * 255).astype(np.uint8)
    
    # Save grayscale ratio image
    ratio_png = os.path.join(output_dir, f"S1_VV_VH_Ratio_{date_str}.png")
    Image.fromarray(ratio_8bit, mode='L').save(ratio_png)
    print(f"  [OK] Saved Ratio Image: {ratio_png}")
    
    # Create colored ratio visualization
    # Low ratio = red (rough), Medium = yellow, High = blue (smooth/water)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color scale: Red -> Yellow -> Green -> Blue
    # This shows ratio gradient
    rgb[:,:,0] = np.clip((1 - ratio_norm) * 255, 0, 255).astype(np.uint8)  # Red for low ratio
    rgb[:,:,1] = np.clip(ratio_norm * (1 - ratio_norm) * 4 * 255, 0, 255).astype(np.uint8)  # Green in middle
    rgb[:,:,2] = np.clip(ratio_norm * 255, 0, 255).astype(np.uint8)  # Blue for high ratio
    
    colored_png = os.path.join(output_dir, f"S1_VV_VH_Ratio_Colored_{date_str}.png")
    Image.fromarray(rgb, mode='RGB').save(colored_png)
    print(f"  [OK] Saved Colored Ratio: {colored_png}")
    
    # Create histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(ratio.flatten(), bins=100, range=(-5, 20), color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=8, color='red', linestyle='--', label='Water threshold (8 dB)')
    axes[0].axvline(x=5, color='orange', linestyle='--', label='Glacier threshold (5 dB)')
    axes[0].set_xlabel('VV/VH Ratio (dB)')
    axes[0].set_ylabel('Pixel Count')
    axes[0].set_title('VV/VH Ratio Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Ratio image
    axes[1].imshow(ratio, cmap='viridis', vmin=r_min, vmax=r_max)
    axes[1].set_title('VV/VH Ratio Map')
    axes[1].axis('off')
    cbar = plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Ratio (dB)')
    
    plt.tight_layout()
    hist_path = os.path.join(output_dir, f"S1_VV_VH_Analysis_{date_str}.png")
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved Analysis Plot: {hist_path}")
    
    return ratio


def classify_by_vv_vh_ratio(vv, vh, ratio, water_threshold=8.0, glacier_threshold=5.0):
    """
    Classify using VV/VH ratio.
    
    Parameters:
        water_threshold: Ratio above this = potential water (default 8 dB)
        glacier_threshold: Ratio above this but below water = glacier (default 5 dB)
    
    Classification logic:
        - Water: VV/VH > water_threshold AND low VV backscatter
        - Glacier: water_threshold > VV/VH > glacier_threshold
        - Non-glacier: VV/VH < glacier_threshold
    """
    h, w = vv.shape
    class_map = np.zeros((h, w), dtype=np.uint8)
    
    # Compute local texture (to confirm water is smooth)
    local_mean = uniform_filter(vv.astype(np.float64), 7)
    local_sqr_mean = uniform_filter(vv.astype(np.float64)**2, 7)
    local_var = np.maximum(local_sqr_mean - local_mean**2, 0)
    cv = np.sqrt(local_var) / (np.abs(local_mean) + 1e-10)
    
    # =========================================================================
    # WATER Detection using VV/VH ratio
    # =========================================================================
    # Water has HIGH ratio (VV >> VH) AND low backscatter AND smooth
    water_high_ratio = ratio > water_threshold
    water_low_backscatter = vv < -12  # Dark areas
    water_smooth = cv < 0.25
    
    water_mask = water_high_ratio & water_low_backscatter & water_smooth
    
    # Clean water
    water_mask = morphology.remove_small_objects(water_mask.astype(bool), min_size=100)
    water_mask = morphology.remove_small_holes(water_mask, area_threshold=50)
    
    # =========================================================================
    # GLACIER Detection
    # =========================================================================
    # Glacier has medium-high ratio
    glacier_ratio = (ratio > glacier_threshold) & (ratio <= 12)
    glacier_brightness = vv > -15
    
    glacier_mask = glacier_ratio & glacier_brightness & (~water_mask)
    
    # Clean glacier
    glacier_mask = morphology.remove_small_objects(glacier_mask.astype(bool), min_size=50)
    glacier_mask = binary_closing(glacier_mask, iterations=2)
    
    # Apply
    class_map[glacier_mask] = 1  # Glacier
    class_map[water_mask] = 2    # Lake
    
    return class_map


def save_outputs(class_map, vv, profile, output_dir, date_str):
    """Save all outputs"""
    h, w = class_map.shape
    
    # GeoTIFF
    tiff_path = os.path.join(output_dir, f"S1_Classification_{date_str}.tif")
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
    with rasterio.open(tiff_path, 'w', **profile) as dst:
        dst.write(class_map.astype(np.uint8), 1)
    print(f"  [OK] Saved GeoTIFF: {tiff_path}")
    
    # Classification PNG
    png_path = os.path.join(output_dir, f"S1_Classification_{date_str}.png")
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[class_map == 0] = [139, 90, 43]    # Brown - Non-Glacier
    rgb[class_map == 1] = [250, 250, 255]  # White - Glacier
    rgb[class_map == 2] = [0, 50, 180]     # Blue - Lake
    Image.fromarray(rgb, mode='RGB').save(png_path)
    print(f"  [OK] Saved PNG: {png_path}")
    
    # Overlay
    overlay_path = os.path.join(output_dir, f"S1_Overlay_{date_str}.png")
    p1, p99 = np.percentile(vv, [1, 99])
    vv_8bit = np.clip((vv - p1) / (p99 - p1 + 1e-10) * 255, 0, 255).astype(np.uint8)
    rgb_overlay = np.stack([vv_8bit, vv_8bit, vv_8bit], axis=-1)
    
    glacier_mask = class_map == 1
    rgb_overlay[glacier_mask, 0] = np.clip(rgb_overlay[glacier_mask, 0] * 0.5 + 100, 0, 255).astype(np.uint8)
    rgb_overlay[glacier_mask, 1] = np.clip(rgb_overlay[glacier_mask, 1] * 0.5 + 200, 0, 255).astype(np.uint8)
    rgb_overlay[glacier_mask, 2] = 255
    
    lake_mask = class_map == 2
    rgb_overlay[lake_mask, 0] = 0
    rgb_overlay[lake_mask, 1] = 50
    rgb_overlay[lake_mask, 2] = 200
    
    Image.fromarray(rgb_overlay, mode='RGB').save(overlay_path)
    print(f"  [OK] Saved Overlay: {overlay_path}")
    
    # GeoJSON
    transform = profile['transform']
    crs = profile['crs']
    
    for class_val, class_name in [(1, 'Glacier'), (2, 'Lake')]:
        mask = (class_map == class_val).astype(np.uint8)
        if np.any(mask):
            results = list(shapes(mask, mask=mask.astype(bool), transform=transform))
            if results:
                geoms = [{'properties': {'class': class_name, 'date': date_str}, 'geometry': s} 
                         for s, v in results]
                gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)
                out_path = os.path.join(output_dir, f"S1_{class_name}_{date_str}.geojson")
                gdf.to_file(out_path, driver='GeoJSON')
                print(f"  [OK] Saved GeoJSON: {out_path}")


def run_vv_vh_ratio_classification(tiff_path, output_dir, water_thresh=8.0, glacier_thresh=5.0):
    """Run VV/VH ratio-based classification"""
    filename = os.path.basename(tiff_path)
    date_str = filename.replace('.tif', '').replace('S1_', '')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"VV/VH RATIO CLASSIFICATION: {date_str}")
    print(f"Water threshold: {water_thresh} dB")
    print(f"Glacier threshold: {glacier_thresh} dB")
    print("=" * 70)
    
    # Load
    print("\n[1] Loading Sentinel-1 data...")
    vv, vh, profile = load_sentinel1(tiff_path)
    print(f"    Shape: {vv.shape}")
    
    # Denoise
    print("\n[2] Denoising...")
    vv_dn = denoise(vv, size=5)
    vh_dn = denoise(vh, size=5)
    
    # Analyze ratio
    print("\n[3] Analyzing VV/VH ratio...")
    ratio = analyze_vv_vh_ratio(vv_dn, vh_dn, output_dir, date_str)
    
    # Classify
    print("\n[4] Classifying by VV/VH ratio...")
    class_map = classify_by_vv_vh_ratio(vv_dn, vh_dn, ratio, 
                                         water_threshold=water_thresh,
                                         glacier_threshold=glacier_thresh)
    
    # Stats
    total = class_map.size
    lake_pct = (class_map == 2).sum() / total * 100
    glacier_pct = (class_map == 1).sum() / total * 100
    other_pct = (class_map == 0).sum() / total * 100
    
    print(f"\n    Results:")
    print(f"    Non-Glacier: {other_pct:.1f}%")
    print(f"    Glacier:     {glacier_pct:.1f}%")
    print(f"    Lake:        {lake_pct:.1f}%")
    
    # Save
    print("\n[5] Saving outputs...")
    save_outputs(class_map, vv_dn, profile, output_dir, date_str)
    
    print("\n" + "=" * 70)
    print("VV/VH RATIO CLASSIFICATION COMPLETE!")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    return class_map, ratio


if __name__ == "__main__":
    TIFF_PATH = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1\S1_2023-07-18.tif"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_vv_vh_ratio"
    
    # Tune these thresholds:
    # Higher water_thresh = fewer false positives (shadows)
    # Lower water_thresh = more lakes detected (but maybe shadows too)
    run_vv_vh_ratio_classification(TIFF_PATH, OUTPUT_DIR, 
                                    water_thresh=7.5,    # Ratio > 7.5 dB = water candidate
                                    glacier_thresh=5.0)  # Ratio > 5 dB = glacier
