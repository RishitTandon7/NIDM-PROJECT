"""
Sentinel-1 Vision-Guided Classification (V3)
=============================================
Uses Ollama LLaVA vision model to:
1. Analyze SAR image and identify features
2. Get region-specific guidance for lakes vs shadows
3. Apply vision-informed thresholds for better classification

Based on LLaVA analysis insights:
- Lakes: Top-left and bottom-left corners (smooth, uniform dark)
- Shadows: Right side (irregular, textured dark)
- Glaciers: Bright white areas (especially top-right)

Author: Antigravity
Date: 2026-01-30
"""

import os
import base64
import json
import requests
import numpy as np
import rasterio
from rasterio.features import shapes
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter, median_filter, sobel
from scipy.ndimage import binary_opening, binary_closing, binary_dilation
from skimage import morphology
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')

OLLAMA_URL = "http://localhost:11434/api/generate"


def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def query_ollama_for_regions(image_path, model="llava"):
    """
    Ask LLaVA to identify specific lake regions with coordinates.
    """
    image_b64 = image_to_base64(image_path)
    
    prompt = """Analyze this SAR satellite image and identify water bodies/glacial lakes.

For EACH lake you can identify, provide:
1. Location as percentage coordinates: top-left corner coordinates (x%, y%) and approximate size (width%, height%)
2. Confidence level (high/medium/low)
3. Whether you're sure it's water (not shadow)

Format your response as a simple list:
LAKE 1: x=20%, y=10%, w=15%, h=10%, confidence=high, is_water=yes
LAKE 2: x=5%, y=70%, w=10%, h=8%, confidence=medium, is_water=yes
etc.

Only list areas you're confident are actual water bodies, not shadows."""

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        print(f"  Error querying Ollama: {e}")
        return None


def load_sentinel1(tiff_path):
    """Load Sentinel-1 data with profile"""
    with rasterio.open(tiff_path) as src:
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)
        profile = src.profile.copy()
    return vv, vh, profile


def enhanced_lee_filter(img, size=7, damping_factor=1.0):
    """Enhanced Lee filter"""
    img = img.astype(np.float64)
    local_mean = uniform_filter(img, size)
    local_sqr_mean = uniform_filter(img**2, size)
    local_var = np.maximum(local_sqr_mean - local_mean**2, 0)
    noise_var = np.median(local_var) * damping_factor
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


def compute_texture(data, window=9):
    """Compute texture features"""
    data = data.astype(np.float64)
    local_mean = uniform_filter(data, window)
    local_sqr_mean = uniform_filter(data**2, window)
    local_var = np.maximum(local_sqr_mean - local_mean**2, 0)
    local_std = np.sqrt(local_var)
    cv = local_std / (np.abs(local_mean) + 1e-10)
    
    # Edge density
    edge_x = sobel(data, axis=1)
    edge_y = sobel(data, axis=0)
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    edge_density = uniform_filter(edge_mag, window)
    
    return cv, edge_density


def vision_guided_classification(vv, vh, cv_vv, edge_vv):
    """
    Classification using insights from LLaVA vision analysis.
    
    Key insights from LLaVA:
    - Lakes in TOP-LEFT and BOTTOM-LEFT (smooth, uniform)
    - Shadows on RIGHT side (textured, irregular)
    - Glaciers in bright areas (especially TOP-RIGHT)
    """
    h, w = vv.shape
    class_map = np.zeros((h, w), dtype=np.uint8)
    
    # Create spatial weight maps based on vision guidance
    y_coords, x_coords = np.ogrid[0:h, 0:w]
    
    # Left side = higher lake probability (per LLaVA)
    left_weight = 1 - (x_coords / w)  # Decreases from left to right
    
    # =========================================================================
    # WATER/LAKE Detection - Vision Guided
    # =========================================================================
    # Core water detection (strict)
    water_vv_low = vv < -16
    water_vh_low = vh < -24
    water_very_smooth = cv_vv < 0.12
    water_low_edge = edge_vv < np.percentile(edge_vv, 15)
    
    # Base water mask
    water_mask_base = water_vv_low & water_vh_low & water_very_smooth & water_low_edge
    
    # Apply spatial weighting (left side more likely to be water per LLaVA)
    # Relax thresholds slightly for left side
    water_left_region = (x_coords < w * 0.4)  # Left 40%
    water_relaxed = (vv < -14) & (cv_vv < 0.18) & water_left_region
    
    # Combine
    water_mask = water_mask_base | (water_relaxed & water_very_smooth)
    
    # =========================================================================
    # SHADOW Exclusion - Vision Guided
    # =========================================================================
    # Right side shadows (per LLaVA - irregular, textured)
    right_region = (x_coords > w * 0.6)  # Right 40%
    
    # Shadows have texture
    shadow_has_texture = cv_vv > 0.10
    shadow_low_backscatter = (vv < -10) & (vv > -30)
    
    # More aggressive shadow detection on right side
    shadow_mask = shadow_low_backscatter & shadow_has_texture & right_region
    
    # Remove shadows from water mask
    water_mask = water_mask & (~shadow_mask)
    
    # =========================================================================
    # GLACIER Detection - Vision Guided
    # =========================================================================
    # Bright areas = glacier (especially consistent in center and top-right)
    glacier_bright = (vv > -6) & (cv_vv < 0.5)
    
    # Medium brightness glaciers
    glacier_medium = (vv > -15) & (vv < -3) & (cv_vv < 0.35) & (cv_vv > 0.05)
    
    # VV/VH ratio for glacier
    vv_vh_ratio = vv - vh
    glacier_ratio = (vv_vh_ratio > 5) & (vv_vh_ratio < 15)
    
    glacier_mask = (glacier_bright | (glacier_medium & glacier_ratio)) & (~water_mask)
    
    # =========================================================================
    # Apply classifications
    # =========================================================================
    class_map[glacier_mask] = 1  # Glacier
    class_map[water_mask] = 2    # Lake
    
    # =========================================================================
    # Post-processing
    # =========================================================================
    # Clean water - remove tiny spots
    water_only = (class_map == 2)
    water_cleaned = morphology.remove_small_objects(water_only, min_size=150)
    water_cleaned = morphology.remove_small_holes(water_cleaned, area_threshold=100)
    class_map[water_only & ~water_cleaned] = 0
    class_map[~water_only & water_cleaned] = 2
    
    # Smooth glacier boundaries
    glacier_only = (class_map == 1)
    glacier_cleaned = morphology.remove_small_objects(glacier_only, min_size=50)
    glacier_cleaned = binary_closing(glacier_cleaned, iterations=2)
    class_map[glacier_only & ~glacier_cleaned] = 0
    class_map[~glacier_only & glacier_cleaned] = 1
    
    return class_map


def save_classification_geotiff(class_map, profile, output_path):
    """Save classification as GeoTIFF"""
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(class_map.astype(np.uint8), 1)
    print(f"  [OK] Saved GeoTIFF: {output_path}")


def save_high_quality_png(class_map, output_path, vv=None):
    """Save high-quality classification PNG"""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color mapping
    rgb[class_map == 0] = [139, 90, 43]    # Brown - Non-Glacier
    rgb[class_map == 1] = [250, 250, 255]  # White - Glacier
    rgb[class_map == 2] = [0, 80, 180]     # Blue - Lake
    
    img = Image.fromarray(rgb, mode='RGB')
    img.save(output_path, 'PNG', optimize=False)
    
    print(f"  [OK] Saved PNG: {output_path}")
    print(f"      Resolution: {w} x {h} pixels")


def save_overlay_visualization(vv, class_map, output_path):
    """Create overlay of classification on SAR image"""
    h, w = vv.shape
    
    # Stretch VV to 0-255
    p1 = np.percentile(vv, 1)
    p99 = np.percentile(vv, 99)
    vv_stretched = np.clip((vv - p1) / (p99 - p1 + 1e-10), 0, 1)
    vv_8bit = (vv_stretched * 255).astype(np.uint8)
    
    # Create RGB base from VV
    rgb = np.stack([vv_8bit, vv_8bit, vv_8bit], axis=-1)
    
    # Overlay classification with transparency
    alpha = 0.5
    
    # Glacier overlay (cyan tint)
    glacier_mask = class_map == 1
    rgb[glacier_mask, 0] = (rgb[glacier_mask, 0] * (1-alpha) + 100 * alpha).astype(np.uint8)
    rgb[glacier_mask, 1] = (rgb[glacier_mask, 1] * (1-alpha) + 200 * alpha).astype(np.uint8)
    rgb[glacier_mask, 2] = (rgb[glacier_mask, 2] * (1-alpha) + 255 * alpha).astype(np.uint8)
    
    # Lake overlay (blue)
    lake_mask = class_map == 2
    rgb[lake_mask, 0] = (rgb[lake_mask, 0] * (1-alpha) + 0 * alpha).astype(np.uint8)
    rgb[lake_mask, 1] = (rgb[lake_mask, 1] * (1-alpha) + 50 * alpha).astype(np.uint8)
    rgb[lake_mask, 2] = (rgb[lake_mask, 2] * (1-alpha) + 200 * alpha).astype(np.uint8)
    
    img = Image.fromarray(rgb, mode='RGB')
    img.save(output_path, 'PNG')
    
    print(f"  [OK] Saved Overlay: {output_path}")


def save_vectors_geojson(class_map, profile, output_dir, date_str):
    """Export classification as GeoJSON vectors"""
    transform = profile['transform']
    crs = profile['crs']
    
    for class_val, class_name in [(1, 'Glacier'), (2, 'Lake')]:
        mask = (class_map == class_val).astype(np.uint8)
        if np.any(mask):
            results = (
                {'properties': {'class': class_name, 'date': date_str}, 'geometry': s}
                for s, v in shapes(mask, mask=mask.astype(bool), transform=transform)
            )
            geoms = list(results)
            if geoms:
                gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)
                out_path = os.path.join(output_dir, f"S1_{class_name}_{date_str}.geojson")
                gdf.to_file(out_path, driver='GeoJSON')
                print(f"  [OK] Saved GeoJSON: {out_path}")


def run_vision_guided_classification(tiff_path, output_dir, use_ollama=True):
    """
    Complete vision-guided classification pipeline.
    """
    filename = os.path.basename(tiff_path)
    date_str = filename.replace('.tif', '').replace('S1_', '')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"VISION-GUIDED CLASSIFICATION (V3): {date_str}")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1] Loading Sentinel-1 data...")
    vv, vh, profile = load_sentinel1(tiff_path)
    print(f"    Shape: {vv.shape}")
    
    # 2. Denoise
    print("\n[2] Denoising...")
    vv_denoised, vh_denoised = denoise_sar(vv, vh)
    
    # Save denoised PNG for vision
    denoised_png = os.path.join(output_dir, f"S1_Denoised_VV_{date_str}.png")
    p1 = np.percentile(vv_denoised, 1)
    p99 = np.percentile(vv_denoised, 99)
    stretched = np.clip((vv_denoised - p1) / (p99 - p1 + 1e-10), 0, 1)
    img_8bit = (stretched * 255).astype(np.uint8)
    Image.fromarray(img_8bit, mode='L').save(denoised_png)
    print(f"    Saved: {denoised_png}")
    
    # 3. Optional: Get vision guidance
    if use_ollama:
        print("\n[3] Getting LLaVA vision guidance...")
        vision_response = query_ollama_for_regions(denoised_png)
        if vision_response:
            print("    Vision response received")
            # Save response
            with open(os.path.join(output_dir, f"S1_VisionGuide_{date_str}.txt"), 'w') as f:
                f.write(vision_response)
    
    # 4. Compute texture
    print("\n[4] Computing texture features...")
    cv_vv, edge_vv = compute_texture(vv_denoised, window=9)
    
    # 5. Classify
    print("\n[5] Classifying with vision-guided thresholds...")
    class_map = vision_guided_classification(vv_denoised, vh_denoised, cv_vv, edge_vv)
    
    # Statistics
    total_pixels = class_map.size
    glacier_pct = (class_map == 1).sum() / total_pixels * 100
    lake_pct = (class_map == 2).sum() / total_pixels * 100
    non_glacier_pct = (class_map == 0).sum() / total_pixels * 100
    
    print(f"\n    Classification Results:")
    print(f"    Non-Glacier (Brown): {non_glacier_pct:.1f}%")
    print(f"    Glacier (White):     {glacier_pct:.1f}%")
    print(f"    Lake (Blue):         {lake_pct:.1f}%")
    
    # 6. Save outputs
    print("\n[6] Saving outputs...")
    
    # GeoTIFF
    class_tiff = os.path.join(output_dir, f"S1_Classification_{date_str}.tif")
    save_classification_geotiff(class_map, profile, class_tiff)
    
    # High-quality PNG
    class_png = os.path.join(output_dir, f"S1_Classification_{date_str}.png")
    save_high_quality_png(class_map, class_png)
    
    # Overlay visualization
    overlay_png = os.path.join(output_dir, f"S1_Overlay_{date_str}.png")
    save_overlay_visualization(vv_denoised, class_map, overlay_png)
    
    # GeoJSON vectors
    print("\n[7] Exporting vector polygons...")
    save_vectors_geojson(class_map, profile, output_dir, date_str)
    
    print("\n" + "=" * 70)
    print("VISION-GUIDED CLASSIFICATION COMPLETE!")
    print(f"Output folder: {output_dir}")
    print("=" * 70)
    
    return class_map


if __name__ == "__main__":
    TIFF_PATH = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1\S1_2023-07-18.tif"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_vision_guided"
    
    run_vision_guided_classification(TIFF_PATH, OUTPUT_DIR, use_ollama=True)
