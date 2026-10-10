"""
Sentinel-1 Ultra-Strict Lake Detection (V4)
============================================
VERY aggressive shadow filtering:
- Ultra-strict texture threshold (lakes are EXTREMELY smooth)
- Multi-scale texture analysis
- Shape analysis (shadows are elongated, lakes are more compact)
- Edge-based filtering

Author: Antigravity
Date: 2026-01-30
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
from PIL import Image
from scipy.ndimage import uniform_filter, median_filter, sobel, gaussian_filter
from scipy.ndimage import binary_opening, binary_closing, label as ndimage_label
from scipy.ndimage import minimum_filter, maximum_filter
from skimage import morphology
from skimage.measure import regionprops, label as sk_label
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')


def load_sentinel1(tiff_path):
    """Load Sentinel-1 data"""
    with rasterio.open(tiff_path) as src:
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)
        profile = src.profile.copy()
    return vv, vh, profile


def enhanced_lee_filter(img, size=7):
    """Enhanced Lee filter"""
    img = img.astype(np.float64)
    local_mean = uniform_filter(img, size)
    local_sqr_mean = uniform_filter(img**2, size)
    local_var = np.maximum(local_sqr_mean - local_mean**2, 0)
    noise_var = np.median(local_var)
    cu = np.sqrt(noise_var) / (np.abs(local_mean) + 1e-10)
    ci = np.sqrt(local_var) / (np.abs(local_mean) + 1e-10)
    w = 1 - (cu**2 / (ci**2 + 1e-10))
    w = np.clip(w, 0, 1)
    return (local_mean + w * (img - local_mean)).astype(np.float32)


def denoise_sar(vv, vh):
    """Denoise SAR bands"""
    vv_med = median_filter(vv, size=3)
    vh_med = median_filter(vh, size=3)
    return enhanced_lee_filter(vv_med, 5), enhanced_lee_filter(vh_med, 5)


def compute_multi_scale_texture(data):
    """
    Compute texture at multiple scales to better distinguish lakes from shadows.
    Lakes are smooth at ALL scales, shadows may have texture at some scales.
    """
    data = data.astype(np.float64)
    
    textures = {}
    
    for window in [5, 9, 15, 21]:
        # Local statistics
        local_mean = uniform_filter(data, window)
        local_sqr_mean = uniform_filter(data**2, window)
        local_var = np.maximum(local_sqr_mean - local_mean**2, 0)
        local_std = np.sqrt(local_var)
        
        # Coefficient of variation
        cv = local_std / (np.abs(local_mean) + 1e-10)
        
        # Local range
        local_min = minimum_filter(data, size=window)
        local_max = maximum_filter(data, size=window)
        local_range = local_max - local_min
        
        textures[f'cv_{window}'] = cv
        textures[f'range_{window}'] = local_range
        textures[f'std_{window}'] = local_std
    
    # Edge magnitude
    edge_x = sobel(data, axis=1)
    edge_y = sobel(data, axis=0)
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    textures['edge_mag'] = edge_mag
    textures['edge_density'] = uniform_filter(edge_mag, 9)
    
    return textures


def is_water_by_shape(binary_mask, min_solidity=0.5, max_eccentricity=0.95):
    """
    Filter water candidates by shape.
    - Lakes tend to be more compact (higher solidity)
    - Shadows tend to be elongated (higher eccentricity)
    """
    labeled = sk_label(binary_mask)
    
    valid_mask = np.zeros_like(binary_mask)
    
    regions = regionprops(labeled)
    for region in regions:
        # Shape criteria
        solidity = region.solidity if hasattr(region, 'solidity') else 1.0
        eccentricity = region.eccentricity if hasattr(region, 'eccentricity') else 0.0
        area = region.area
        
        # Lakes: reasonably compact, not too elongated
        # Shadows: often elongated (high eccentricity, low solidity)
        if (solidity > min_solidity and 
            eccentricity < max_eccentricity and 
            area > 100):
            valid_mask[labeled == region.label] = True
    
    return valid_mask


def ultra_strict_water_detection(vv, vh, textures):
    """
    Ultra-strict water detection using multiple criteria.
    
    Real water characteristics:
    1. VERY low backscatter (VV < -18 dB for calm water)
    2. EXTREMELY low texture at ALL scales
    3. VERY low edge density
    4. Relatively compact shape (not elongated like shadows)
    """
    h, w = vv.shape
    
    # Extract textures
    cv_5 = textures['cv_5']
    cv_9 = textures['cv_9']
    cv_15 = textures['cv_15']
    cv_21 = textures['cv_21']
    
    range_9 = textures['range_9']
    range_15 = textures['range_15']
    
    edge_density = textures['edge_density']
    edge_mag = textures['edge_mag']
    
    # =========================================================================
    # ULTRA-STRICT WATER CRITERIA
    # =========================================================================
    
    # 1. Very low backscatter
    water_vv_low = vv < -18  # Stricter than before
    water_vh_low = vh < -26  # Stricter than before
    
    # 2. ULTRA-LOW texture at ALL scales
    # Real lakes have CV < 0.05, shadows typically have CV > 0.1
    water_ultra_smooth_5 = cv_5 < 0.08
    water_ultra_smooth_9 = cv_9 < 0.08
    water_ultra_smooth_15 = cv_15 < 0.10
    water_ultra_smooth_21 = cv_21 < 0.12
    
    # All scales must be smooth
    water_all_scales_smooth = (water_ultra_smooth_5 & water_ultra_smooth_9 & 
                                water_ultra_smooth_15 & water_ultra_smooth_21)
    
    # 3. Very low local range
    water_low_range = range_9 < np.percentile(range_9, 10)
    
    # 4. Very low edge density
    water_no_edges = edge_density < np.percentile(edge_density, 8)
    
    # Combine all criteria
    water_mask = (water_vv_low & water_vh_low & 
                  water_all_scales_smooth & 
                  water_low_range & 
                  water_no_edges)
    
    # 5. Shape filtering - remove elongated regions (likely shadows)
    water_mask = is_water_by_shape(water_mask, min_solidity=0.4, max_eccentricity=0.92)
    
    # 6. Remove small water bodies (noise)
    water_mask = morphology.remove_small_objects(water_mask.astype(bool), min_size=200)
    water_mask = morphology.remove_small_holes(water_mask, area_threshold=100)
    
    return water_mask


def classify_sar_ultra_strict(vv, vh, textures):
    """
    Ultra-strict classification with aggressive shadow filtering.
    """
    h, w = vv.shape
    class_map = np.zeros((h, w), dtype=np.uint8)
    
    # Detect water with ultra-strict criteria
    water_mask = ultra_strict_water_detection(vv, vh, textures)
    
    # Glacier detection (unchanged)
    cv_9 = textures['cv_9']
    vv_vh_ratio = vv - vh
    
    glacier_bright = (vv > -6) & (cv_9 < 0.5)
    glacier_medium = (vv > -14) & (vv < -3) & (cv_9 < 0.35)
    glacier_ratio = (vv_vh_ratio > 5) & (vv_vh_ratio < 15)
    
    glacier_mask = (glacier_bright | (glacier_medium & glacier_ratio)) & (~water_mask)
    
    # Clean glacier
    glacier_mask = morphology.remove_small_objects(glacier_mask.astype(bool), min_size=50)
    glacier_mask = binary_closing(glacier_mask, iterations=2)
    
    # Apply classifications
    class_map[glacier_mask] = 1  # Glacier
    class_map[water_mask] = 2    # Lake
    
    return class_map


def save_outputs(class_map, vv, profile, output_dir, date_str):
    """Save all outputs"""
    os.makedirs(output_dir, exist_ok=True)
    
    h, w = class_map.shape
    
    # 1. GeoTIFF
    tiff_path = os.path.join(output_dir, f"S1_Classification_{date_str}.tif")
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
    with rasterio.open(tiff_path, 'w', **profile) as dst:
        dst.write(class_map.astype(np.uint8), 1)
    print(f"  [OK] Saved GeoTIFF: {tiff_path}")
    
    # 2. Classification PNG
    png_path = os.path.join(output_dir, f"S1_Classification_{date_str}.png")
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[class_map == 0] = [139, 90, 43]    # Brown
    rgb[class_map == 1] = [250, 250, 255]  # White
    rgb[class_map == 2] = [0, 50, 180]     # Dark Blue
    Image.fromarray(rgb, mode='RGB').save(png_path)
    print(f"  [OK] Saved PNG: {png_path}")
    
    # 3. Overlay
    overlay_path = os.path.join(output_dir, f"S1_Overlay_{date_str}.png")
    p1, p99 = np.percentile(vv, [1, 99])
    vv_8bit = np.clip((vv - p1) / (p99 - p1 + 1e-10) * 255, 0, 255).astype(np.uint8)
    rgb_overlay = np.stack([vv_8bit, vv_8bit, vv_8bit], axis=-1)
    
    # Light cyan for glacier
    glacier_mask = class_map == 1
    rgb_overlay[glacier_mask, 0] = np.clip(rgb_overlay[glacier_mask, 0] * 0.5 + 100, 0, 255).astype(np.uint8)
    rgb_overlay[glacier_mask, 1] = np.clip(rgb_overlay[glacier_mask, 1] * 0.5 + 200, 0, 255).astype(np.uint8)
    rgb_overlay[glacier_mask, 2] = 255
    
    # Strong blue for lake
    lake_mask = class_map == 2
    rgb_overlay[lake_mask, 0] = 0
    rgb_overlay[lake_mask, 1] = 50
    rgb_overlay[lake_mask, 2] = 200
    
    Image.fromarray(rgb_overlay, mode='RGB').save(overlay_path)
    print(f"  [OK] Saved Overlay: {overlay_path}")
    
    # 4. Denoised VV
    denoised_path = os.path.join(output_dir, f"S1_Denoised_VV_{date_str}.png")
    Image.fromarray(vv_8bit, mode='L').save(denoised_path)
    print(f"  [OK] Saved Denoised: {denoised_path}")
    
    # 5. GeoJSON vectors
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


def run_ultra_strict_classification(tiff_path, output_dir):
    """Run ultra-strict classification"""
    filename = os.path.basename(tiff_path)
    date_str = filename.replace('.tif', '').replace('S1_', '')
    
    print("=" * 70)
    print(f"ULTRA-STRICT CLASSIFICATION (V4): {date_str}")
    print("Aggressive shadow filtering enabled")
    print("=" * 70)
    
    # Load
    print("\n[1] Loading...")
    vv, vh, profile = load_sentinel1(tiff_path)
    print(f"    Shape: {vv.shape}")
    
    # Denoise
    print("\n[2] Denoising...")
    vv_dn, vh_dn = denoise_sar(vv, vh)
    
    # Multi-scale texture
    print("\n[3] Computing multi-scale texture...")
    textures = compute_multi_scale_texture(vv_dn)
    
    # Classify
    print("\n[4] Classifying with ULTRA-STRICT lake detection...")
    class_map = classify_sar_ultra_strict(vv_dn, vh_dn, textures)
    
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
    print("ULTRA-STRICT CLASSIFICATION COMPLETE!")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    return class_map


if __name__ == "__main__":
    TIFF_PATH = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1\S1_2023-07-18.tif"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_ultra_strict"
    
    run_ultra_strict_classification(TIFF_PATH, OUTPUT_DIR)
