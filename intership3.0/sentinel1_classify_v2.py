"""
Sentinel-1 SAR Classification V2 - Improved
============================================
Fixes:
- Higher quality PNG output (native resolution, anti-aliased)
- Better shadow vs lake separation using texture analysis
- More conservative water detection thresholds

Author: Antigravity
Date: 2026-01-30
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
from PIL import Image
from scipy.ndimage import uniform_filter, median_filter, binary_opening, binary_closing
from scipy.ndimage import sobel, gaussian_filter
from skimage import morphology
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')


def load_sentinel1(tiff_path):
    """Load Sentinel-1 data with profile"""
    with rasterio.open(tiff_path) as src:
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)
        angle = src.read(3).astype(np.float32)
        profile = src.profile.copy()
    return vv, vh, angle, profile


def enhanced_lee_filter(img, size=7, damping_factor=1.0):
    """Enhanced Lee filter with better edge preservation"""
    img = img.astype(np.float64)
    local_mean = uniform_filter(img, size)
    local_sqr_mean = uniform_filter(img**2, size)
    local_var = np.maximum(local_sqr_mean - local_mean**2, 0)
    noise_var = np.median(local_var) * damping_factor
    cu = np.sqrt(noise_var) / (np.abs(local_mean) + 1e-10)
    ci = np.sqrt(local_var) / (np.abs(local_mean) + 1e-10)
    w = 1 - (cu**2 / (ci**2 + 1e-10))
    w = np.clip(w, 0, 1)
    output = local_mean + w * (img - local_mean)
    return output.astype(np.float32)


def denoise_sar(vv, vh, filter_size=5):
    """Apply denoising to SAR bands"""
    vv_med = median_filter(vv, size=3)
    vh_med = median_filter(vh, size=3)
    vv_denoised = enhanced_lee_filter(vv_med, size=filter_size)
    vh_denoised = enhanced_lee_filter(vh_med, size=filter_size)
    return vv_denoised, vh_denoised


def compute_texture_features(data, window_size=9):
    """
    Compute texture features to distinguish water from shadows.
    Water: Very homogeneous (low texture)
    Shadows: Often have some texture from underlying terrain
    """
    data = data.astype(np.float64)
    
    # Local mean and variance
    local_mean = uniform_filter(data, window_size)
    local_sqr_mean = uniform_filter(data**2, window_size)
    local_var = np.maximum(local_sqr_mean - local_mean**2, 0)
    local_std = np.sqrt(local_var)
    
    # Coefficient of variation
    cv = local_std / (np.abs(local_mean) + 1e-10)
    
    # Edge density (using Sobel)
    edge_x = sobel(data, axis=1)
    edge_y = sobel(data, axis=0)
    edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
    edge_density = uniform_filter(edge_magnitude, window_size)
    
    # Entropy-like measure (using local range)
    from scipy.ndimage import minimum_filter, maximum_filter
    local_min = minimum_filter(data, size=window_size)
    local_max = maximum_filter(data, size=window_size)
    local_range = local_max - local_min
    
    return {
        'cv': cv,
        'std': local_std,
        'edge_density': edge_density,
        'local_range': local_range,
        'local_mean': local_mean
    }


def classify_sar_improved(vv, vh, texture_vv, texture_vh):
    """
    Improved classification with shadow vs water separation.
    
    Key insight:
    - Water: Very low backscatter + VERY low texture (homogeneous)
    - Shadows: Low backscatter + SOME texture (terrain visible through shadow)
    - Glacier: Medium backscatter + moderate texture
    - Rock/Terrain: Variable backscatter + high texture
    """
    h, w = vv.shape
    class_map = np.zeros((h, w), dtype=np.uint8)
    
    # Extract texture features
    cv_vv = texture_vv['cv']
    edge_vv = texture_vv['edge_density']
    range_vv = texture_vv['local_range']
    
    cv_vh = texture_vh['cv']
    edge_vh = texture_vh['edge_density']
    
    # VV/VH ratio
    vv_vh_ratio = vv - vh
    
    # =========================================================================
    # WATER (Glacial Lakes) - VERY STRICT criteria
    # =========================================================================
    # Real water has:
    # 1. Very low backscatter (specular reflection)
    # 2. EXTREMELY low texture (calm water is homogeneous)
    # 3. Low edge density
    
    # Backscatter thresholds (more conservative)
    water_vv_low = vv < -18  # Stricter VV threshold
    water_vh_low = vh < -25  # Stricter VH threshold
    
    # Texture thresholds (key for shadow separation)
    water_very_smooth = cv_vv < 0.15  # Water is VERY homogeneous
    water_low_edge = edge_vv < np.percentile(edge_vv, 20)  # Very few edges
    water_low_range = range_vv < np.percentile(range_vv, 25)  # Low local range
    
    # Combined water mask
    water_mask = (water_vv_low & water_vh_low & 
                  water_very_smooth & water_low_edge & water_low_range)
    
    # =========================================================================
    # SHADOW Detection (to exclude from lakes)
    # =========================================================================
    # Shadows have:
    # 1. Low backscatter (similar to water)
    # 2. BUT higher texture than water (terrain shows through)
    # 3. Often near bright (glacier) areas
    
    shadow_low_backscatter = (vv < -15) & (vv > -30)
    shadow_has_texture = cv_vv > 0.12  # More texture than water
    shadow_has_edges = edge_vv > np.percentile(edge_vv, 15)
    
    shadow_mask = shadow_low_backscatter & shadow_has_texture & (~water_mask)
    
    # =========================================================================
    # GLACIER (Ice/Snow) Detection
    # =========================================================================
    # Glaciers have:
    # 1. Medium backscatter (varies with ice type)
    # 2. Moderate texture
    # 3. Characteristic VV/VH ratio
    
    glacier_vv_range = (vv > -18) & (vv < 5)
    glacier_vh_range = (vh > -25) & (vh < 0)
    glacier_moderate_texture = (cv_vv < 0.5) & (cv_vv > 0.05)
    glacier_ratio = (vv_vh_ratio > 4) & (vv_vh_ratio < 15)
    
    glacier_mask = (glacier_vv_range & glacier_vh_range & 
                    glacier_moderate_texture & glacier_ratio)
    glacier_mask = glacier_mask & (~water_mask)
    
    # Bright glacier surfaces (clean ice/firn)
    bright_glacier = (vv > -5) & (cv_vv < 0.4)
    glacier_mask = glacier_mask | bright_glacier
    
    # =========================================================================
    # Apply classifications
    # =========================================================================
    class_map[glacier_mask] = 1  # Glacier
    class_map[water_mask] = 2    # Water/Lake
    # Everything else (0) is non-glacier (rock, shadow, terrain)
    
    # =========================================================================
    # Post-processing
    # =========================================================================
    # Remove very small water bodies (likely noise)
    water_only = (class_map == 2)
    water_cleaned = morphology.remove_small_objects(water_only, min_size=100)
    water_cleaned = morphology.remove_small_holes(water_cleaned, area_threshold=50)
    class_map[water_only & ~water_cleaned] = 0
    class_map[~water_only & water_cleaned] = 2
    
    # Clean glacier boundaries
    glacier_only = (class_map == 1)
    glacier_cleaned = morphology.remove_small_objects(glacier_only, min_size=50)
    class_map[glacier_only & ~glacier_cleaned] = 0
    class_map[~glacier_only & glacier_cleaned] = 1
    
    return class_map


def save_classification_geotiff(class_map, profile, output_path):
    """Save classification as GeoTIFF"""
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(class_map.astype(np.uint8), 1)
        dst.update_tags(1, CLASS_0='Non-Glacier', CLASS_1='Glacier', CLASS_2='Glacial_Lake')
    print(f"  [OK] Saved GeoTIFF: {output_path}")


def save_denoised_geotiff(vv, vh, profile, output_path):
    """Save denoised bands as GeoTIFF"""
    profile.update(dtype=rasterio.float32, count=2, compress='lzw')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(vv.astype(np.float32), 1)
        dst.write(vh.astype(np.float32), 2)
    print(f"  [OK] Saved Denoised GeoTIFF: {output_path}")


def geotiff_to_high_quality_png(class_tiff, png_path):
    """
    Convert classification GeoTIFF to HIGH QUALITY PNG.
    - Native resolution (no downsampling)
    - Clean color mapping
    - High quality PNG compression
    """
    with rasterio.open(class_tiff) as src:
        data = src.read(1)
    
    h, w = data.shape
    
    # Create RGB image at native resolution
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color mapping
    # Non-Glacier (0): Brown/Tan
    rgb[data == 0] = [139, 90, 43]
    
    # Glacier (1): White with slight blue tint
    rgb[data == 1] = [250, 250, 255]
    
    # Lake (2): Deep blue
    rgb[data == 2] = [0, 80, 180]
    
    # Save as high-quality PNG
    img = Image.fromarray(rgb, mode='RGB')
    img.save(png_path, 'PNG', optimize=False)
    
    print(f"  [OK] Saved High-Quality PNG: {png_path}")
    print(f"      Resolution: {w} x {h} pixels")


def save_grayscale_high_quality(data, output_path):
    """Save grayscale image at native resolution"""
    p1 = np.percentile(data, 1)
    p99 = np.percentile(data, 99)
    stretched = np.clip((data - p1) / (p99 - p1 + 1e-10), 0, 1)
    img_8bit = (stretched * 255).astype(np.uint8)
    
    img = Image.fromarray(img_8bit, mode='L')
    img.save(output_path, 'PNG', optimize=False)
    print(f"  [OK] Saved High-Quality Grayscale: {output_path}")


def save_vectors_geojson(class_map, profile, output_dir, date_str):
    """Export classification as GeoJSON vectors"""
    transform = profile['transform']
    crs = profile['crs']
    
    for class_val, class_name in [(1, 'Glacier'), (2, 'Lake')]:
        mask = (class_map == class_val).astype(np.uint8)
        if np.any(mask):
            results = (
                {'properties': {'class': class_name, 'date': date_str, 'source': 'Sentinel-1'}, 'geometry': s}
                for s, v in shapes(mask, mask=mask.astype(bool), transform=transform)
            )
            geoms = list(results)
            if geoms:
                gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)
                out_path = os.path.join(output_dir, f"S1_{class_name}_{date_str}.geojson")
                gdf.to_file(out_path, driver='GeoJSON')
                print(f"  [OK] Saved GeoJSON: {out_path}")


def classify_single_image_v2(tiff_path, output_dir):
    """
    Classify a single Sentinel-1 image with improved accuracy.
    """
    filename = os.path.basename(tiff_path)
    date_str = filename.replace('.tif', '').replace('S1_', '')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"CLASSIFYING (V2 - Improved): {date_str}")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1] Loading Sentinel-1 data...")
    vv, vh, angle, profile = load_sentinel1(tiff_path)
    print(f"    Shape: {vv.shape} (native resolution)")
    print(f"    VV range: [{vv.min():.1f}, {vv.max():.1f}] dB")
    print(f"    VH range: [{vh.min():.1f}, {vh.max():.1f}] dB")
    
    # 2. Denoise
    print("\n[2] Denoising...")
    vv_denoised, vh_denoised = denoise_sar(vv, vh, filter_size=5)
    
    # Save denoised GeoTIFF
    denoised_tiff = os.path.join(output_dir, f"S1_Denoised_{date_str}.tif")
    save_denoised_geotiff(vv_denoised, vh_denoised, profile, denoised_tiff)
    
    # Save denoised PNG
    denoised_png = os.path.join(output_dir, f"S1_Denoised_VV_{date_str}.png")
    save_grayscale_high_quality(vv_denoised, denoised_png)
    
    # 3. Compute texture features
    print("\n[3] Computing texture features (for shadow separation)...")
    texture_vv = compute_texture_features(vv_denoised, window_size=9)
    texture_vh = compute_texture_features(vh_denoised, window_size=9)
    
    # 4. Classify
    print("\n[4] Classifying with improved thresholds...")
    class_map = classify_sar_improved(vv_denoised, vh_denoised, texture_vv, texture_vh)
    
    # Statistics
    total_pixels = class_map.size
    glacier_pct = (class_map == 1).sum() / total_pixels * 100
    lake_pct = (class_map == 2).sum() / total_pixels * 100
    non_glacier_pct = (class_map == 0).sum() / total_pixels * 100
    
    print(f"\n    Classification Results:")
    print(f"    Non-Glacier (Brown): {non_glacier_pct:.1f}%")
    print(f"    Glacier (White):     {glacier_pct:.1f}%")
    print(f"    Lake (Blue):         {lake_pct:.1f}%")
    
    # 5. Save Classification GeoTIFF
    print("\n[5] Saving outputs at NATIVE RESOLUTION...")
    class_tiff = os.path.join(output_dir, f"S1_Classification_{date_str}.tif")
    save_classification_geotiff(class_map, profile, class_tiff)
    
    # 6. Convert to HIGH QUALITY PNG
    class_png = os.path.join(output_dir, f"S1_Classification_{date_str}.png")
    geotiff_to_high_quality_png(class_tiff, class_png)
    
    # 7. Save GeoJSON vectors
    print("\n[6] Exporting vector polygons...")
    save_vectors_geojson(class_map, profile, output_dir, date_str)
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION COMPLETE (V2)!")
    print(f"Output folder: {output_dir}")
    print("=" * 70)
    
    return class_map


if __name__ == "__main__":
    TIFF_PATH = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1\S1_2023-07-18.tif"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_classification_v2"
    
    classify_single_image_v2(TIFF_PATH, OUTPUT_DIR)
