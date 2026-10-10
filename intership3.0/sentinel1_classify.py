"""
Sentinel-1 SAR Classification
==============================
Classifies SAR data into 3 classes:
  - Class 0: Non-Glacier (Terrain/Rock)
  - Class 1: Glacier (Ice/Snow)
  - Class 2: Glacial Lake (Water)

Saves as GeoTIFF (with georeferencing) then converts to PNG.

Author: Antigravity
Date: 2026-01-30
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
from PIL import Image
from scipy.ndimage import uniform_filter, median_filter, binary_opening, binary_closing
from scipy.ndimage import label as ndimage_label
from skimage import morphology
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')


def load_sentinel1(tiff_path):
    """Load Sentinel-1 data with profile"""
    with rasterio.open(tiff_path) as src:
        vv = src.read(1).astype(np.float32)  # VV (dB)
        vh = src.read(2).astype(np.float32)  # VH (dB)
        angle = src.read(3).astype(np.float32)  # Incidence angle
        profile = src.profile.copy()
    return vv, vh, angle, profile


def lee_filter(img, size=5):
    """Lee speckle filter"""
    img = img.astype(np.float64)
    img_mean = uniform_filter(img, size)
    img_sqr_mean = uniform_filter(img**2, size)
    img_variance = np.maximum(img_sqr_mean - img_mean**2, 0)
    overall_variance = np.var(img)
    weights = img_variance / (img_variance + overall_variance + 1e-10)
    output = img_mean + weights * (img - img_mean)
    return output.astype(np.float32)


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
    # Median first for outliers
    vv_med = median_filter(vv, size=3)
    vh_med = median_filter(vh, size=3)
    
    # Enhanced Lee for adaptive smoothing
    vv_denoised = enhanced_lee_filter(vv_med, size=filter_size)
    vh_denoised = enhanced_lee_filter(vh_med, size=filter_size)
    
    return vv_denoised, vh_denoised


def compute_sar_features(vv, vh):
    """
    Compute SAR features for classification.
    VV, VH are already in dB scale.
    """
    eps = 1e-10
    
    # VV/VH ratio (in dB, this is subtraction)
    vv_vh_ratio = vv - vh
    
    # Cross-polarization difference
    cross_pol = vh - vv
    
    # Local texture (coefficient of variation)
    vv_mean = uniform_filter(vv, size=7)
    vv_sqr_mean = uniform_filter(vv**2, size=7)
    vv_var = np.maximum(vv_sqr_mean - vv_mean**2, 0)
    vv_texture = np.sqrt(vv_var) / (np.abs(vv_mean) + eps)
    
    vh_mean = uniform_filter(vh, size=7)
    vh_sqr_mean = uniform_filter(vh**2, size=7)
    vh_var = np.maximum(vh_sqr_mean - vh_mean**2, 0)
    vh_texture = np.sqrt(vh_var) / (np.abs(vh_mean) + eps)
    
    return {
        'vv': vv,
        'vh': vh,
        'vv_vh_ratio': vv_vh_ratio,
        'cross_pol': cross_pol,
        'vv_texture': vv_texture,
        'vh_texture': vh_texture,
        'vv_mean': vv_mean,
        'vh_mean': vh_mean
    }


def classify_sar(vv, vh, features):
    """
    Rule-based classification using SAR backscatter characteristics.
    
    Water (Lakes): 
        - Very low backscatter (specular reflection)
        - VV < -18 dB, VH < -24 dB typically
        - Low texture (smooth surface)
    
    Glacier (Ice/Snow):
        - Medium-high backscatter
        - Varies with ice type (wet vs dry)
        - Moderate texture
    
    Non-Glacier (Terrain):
        - Variable backscatter
        - High texture (rough surface)
    """
    h, w = vv.shape
    class_map = np.zeros((h, w), dtype=np.uint8)
    
    # Extract features
    vv_vh_ratio = features['vv_vh_ratio']
    vv_texture = features['vv_texture']
    vh_texture = features['vh_texture']
    vv_mean = features['vv_mean']
    vh_mean = features['vh_mean']
    
    # =========================================================================
    # WATER (Glacial Lakes) Detection
    # =========================================================================
    # Water has very low backscatter due to specular reflection
    # In dB: VV typically < -15 dB, VH < -22 dB for calm water
    
    water_vv_low = vv < -15  # Low VV backscatter
    water_vh_low = vh < -22  # Low VH backscatter
    water_smooth = (vv_texture < 0.3) & (vh_texture < 0.4)  # Low texture
    water_ratio = vv_vh_ratio > 4  # VV stronger than VH for water
    
    water_mask = water_vv_low & water_vh_low & water_smooth
    
    # =========================================================================
    # GLACIER (Ice/Snow) Detection
    # =========================================================================
    # Glaciers have characteristic backscatter patterns
    # Dry snow: Low backscatter
    # Wet snow/ice: Medium backscatter
    # Rough ice: Higher backscatter
    
    # Summer glaciers (July) - expect mixed wet/dry conditions
    glacier_vv_range = (vv > -20) & (vv < 0)  # Medium VV range
    glacier_vh_range = (vh > -28) & (vh < -5)  # Medium VH range
    glacier_moderate_texture = (vv_texture < 0.6)  # Not too rough
    glacier_ratio = (vv_vh_ratio > 5) & (vv_vh_ratio < 15)  # Typical glacier ratio
    
    # Bright areas (clean ice/firn)
    bright_glacier = (vv > -8) & (vv < 5) & glacier_moderate_texture
    
    glacier_mask = (glacier_vv_range & glacier_ratio & glacier_moderate_texture) | bright_glacier
    glacier_mask = glacier_mask & (~water_mask)  # Exclude water
    
    # =========================================================================
    # Apply classifications
    # =========================================================================
    # Default is non-glacier (0)
    class_map[glacier_mask] = 1  # Glacier
    class_map[water_mask] = 2    # Water/Lake
    
    # =========================================================================
    # Post-processing: Clean up
    # =========================================================================
    # Remove small isolated regions
    for cls in [1, 2]:
        mask = (class_map == cls)
        # Remove small objects
        cleaned = morphology.remove_small_objects(mask, min_size=50)
        # Fill small holes
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=30)
        class_map[mask & ~cleaned] = 0
        class_map[~mask & cleaned] = cls
    
    # Morphological closing to smooth boundaries
    for cls in [1, 2]:
        mask = (class_map == cls)
        mask = binary_closing(mask, iterations=2)
        mask = binary_opening(mask, iterations=1)
        class_map[class_map == cls] = 0
        class_map[mask] = cls
    
    return class_map


def save_classification_geotiff(class_map, profile, output_path):
    """
    Save classification as GeoTIFF with proper georeferencing.
    """
    # Update profile for single-band uint8 output
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'
    )
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(class_map.astype(np.uint8), 1)
        
        # Add color interpretation
        dst.update_tags(1, CLASS_0='Non-Glacier', CLASS_1='Glacier', CLASS_2='Glacial_Lake')
    
    print(f"  [OK] Saved GeoTIFF: {output_path}")
    return output_path


def save_denoised_geotiff(vv, vh, profile, output_path):
    """Save denoised VV and VH as multi-band GeoTIFF"""
    profile.update(
        dtype=rasterio.float32,
        count=2,
        compress='lzw'
    )
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(vv.astype(np.float32), 1)
        dst.write(vh.astype(np.float32), 2)
        dst.update_tags(1, BAND='VV_Denoised_dB')
        dst.update_tags(2, BAND='VH_Denoised_dB')
    
    print(f"  [OK] Saved Denoised GeoTIFF: {output_path}")
    return output_path


def geotiff_to_png(tiff_path, png_path, colormap=None):
    """
    Convert GeoTIFF to PNG with optional colormap.
    """
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
    
    if colormap is not None:
        # Apply colormap for classification
        h, w = data.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_val, color in colormap.items():
            mask = data == class_val
            rgb[mask] = color
        
        img = Image.fromarray(rgb, mode='RGB')
    else:
        # Grayscale stretch
        p2 = np.percentile(data, 2)
        p98 = np.percentile(data, 98)
        stretched = np.clip((data - p2) / (p98 - p2 + 1e-10), 0, 1)
        img_8bit = (stretched * 255).astype(np.uint8)
        img = Image.fromarray(img_8bit, mode='L')
    
    img.save(png_path, 'PNG')
    print(f"  [OK] Saved PNG: {png_path}")
    return png_path


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


def classify_single_image(tiff_path, output_dir):
    """
    Classify a single Sentinel-1 image.
    Saves: GeoTIFF (classification + denoised), PNG, GeoJSON vectors
    """
    filename = os.path.basename(tiff_path)
    date_str = filename.replace('.tif', '').replace('S1_', '')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"CLASSIFYING: {date_str}")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1] Loading Sentinel-1 data...")
    vv, vh, angle, profile = load_sentinel1(tiff_path)
    print(f"    Shape: {vv.shape}")
    print(f"    VV range: [{vv.min():.1f}, {vv.max():.1f}] dB")
    print(f"    VH range: [{vh.min():.1f}, {vh.max():.1f}] dB")
    
    # 2. Denoise
    print("\n[2] Denoising (Enhanced Lee Filter)...")
    vv_denoised, vh_denoised = denoise_sar(vv, vh, filter_size=5)
    
    # Save denoised GeoTIFF
    denoised_tiff = os.path.join(output_dir, f"S1_Denoised_{date_str}.tif")
    save_denoised_geotiff(vv_denoised, vh_denoised, profile, denoised_tiff)
    
    # Convert denoised to PNG
    denoised_png = os.path.join(output_dir, f"S1_Denoised_VV_{date_str}.png")
    
    # Manual stretch for denoised VV
    p2 = np.percentile(vv_denoised, 1)
    p98 = np.percentile(vv_denoised, 99)
    stretched = np.clip((vv_denoised - p2) / (p98 - p2 + 1e-10), 0, 1)
    img_8bit = (stretched * 255).astype(np.uint8)
    Image.fromarray(img_8bit, mode='L').save(denoised_png)
    print(f"  [OK] Saved PNG: {denoised_png}")
    
    # 3. Compute features
    print("\n[3] Computing SAR features...")
    features = compute_sar_features(vv_denoised, vh_denoised)
    
    # 4. Classify
    print("\n[4] Classifying (Glacier / Lake / Non-Glacier)...")
    class_map = classify_sar(vv_denoised, vh_denoised, features)
    
    # Statistics
    total_pixels = class_map.size
    glacier_pct = (class_map == 1).sum() / total_pixels * 100
    lake_pct = (class_map == 2).sum() / total_pixels * 100
    non_glacier_pct = (class_map == 0).sum() / total_pixels * 100
    
    print(f"    Non-Glacier: {non_glacier_pct:.1f}%")
    print(f"    Glacier:     {glacier_pct:.1f}%")
    print(f"    Lake:        {lake_pct:.1f}%")
    
    # 5. Save Classification GeoTIFF
    print("\n[5] Saving outputs...")
    class_tiff = os.path.join(output_dir, f"S1_Classification_{date_str}.tif")
    save_classification_geotiff(class_map, profile, class_tiff)
    
    # 6. Convert to PNG with colormap
    class_png = os.path.join(output_dir, f"S1_Classification_{date_str}.png")
    colormap = {
        0: [139, 90, 43],    # Brown - Non-Glacier
        1: [255, 255, 255],  # White - Glacier
        2: [0, 100, 200]     # Blue - Lake
    }
    geotiff_to_png(class_tiff, class_png, colormap=colormap)
    
    # 7. Save GeoJSON vectors
    print("\n[6] Exporting vector polygons...")
    save_vectors_geojson(class_map, profile, output_dir, date_str)
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION COMPLETE!")
    print(f"Output folder: {output_dir}")
    print("=" * 70)
    
    return class_map


if __name__ == "__main__":
    # Single image classification
    TIFF_PATH = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1\S1_2023-07-18.tif"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_classification"
    
    classify_single_image(TIFF_PATH, OUTPUT_DIR)
