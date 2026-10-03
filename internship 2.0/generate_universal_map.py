"""
Final Consistent Glacier Classifier (Universal Logic)
Logic: 
1. Shadow Detection: Low Brightness + High Blue/Red ratio (Shadows are blue-tinted darks).
2. Glacier Detection: High Brightness + Low Vegetation.
3. Lake Detection: High NDWI + Dark NIR.
Applies IDENTICAL logic to all 66 images to ensure consistency.
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob
from scipy.ndimage import median_filter

def classify_universal(vrt_path):
    with rasterio.open(vrt_path) as src:
        # Universal Band Mapping (Works for 4, 10, or 12 band files)
        # We assume standard order 1,2,3,4 for Blue,Green,Red,NIR if 4 bands
        # Else we grab 2,3,4,8
        
        try:
            if src.count == 4:
                blue  = src.read(1).astype(float)
                green = src.read(2).astype(float)
                red   = src.read(3).astype(float)
                nir   = src.read(4).astype(float)
            else:
                blue  = src.read(2).astype(float)
                green = src.read(3).astype(float)
                red   = src.read(4).astype(float)
                nir   = src.read(8).astype(float)
        except:
             # Very fallback
            blue  = src.read(1).astype(float)
            green = src.read(2).astype(float)
            red   = src.read(3).astype(float)
            nir   = src.read(4).astype(float)

        transform = src.transform
        crs = src.crs

    # --- INDICES ---
    np.seterr(divide='ignore', invalid='ignore')
    
    brightness = (blue + green + red) / 3.0
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)
    
    # Shadow Index (CSI style approximation): Shadows are dark but have Rayleigh scattering (Blue)
    # Ratios help normalize vs lighting changes
    blue_red_ratio = blue / (red + 1e-6)

    # --- THRESHOLDS ---
    # Determine scale (0-1 vs 0-10000)
    # --- THRESHOLDS ---
    # Determine scale (0-1 vs 0-10000)
    # If max value is > 2000, it's definitely DN (0-10000 or 0-65535)
    # If max value is < 2000 (often < 1.0 or < 255), it's Float/Byte
    
    # Calculate Max Blue safely
    MAX_VAL = np.nanmax(blue)
    
    # Defaults
    bright_thresh = 2200
    shadow_bright_thresh = 1500 
    nir_thresh = 1500
    nir_water_thresh = 1500 # Used for water detection logic

    # Check for Float Scale (0-1)
    if MAX_VAL < 100: 
        is_dn = False
        bright_thresh = 0.22
        shadow_bright_thresh = 0.15
        nir_thresh = 0.15
        nir_water_thresh = 0.2
    else:
        is_dn = True
        bright_thresh = 2200
        shadow_bright_thresh = 1500
        nir_thresh = 1500
        # nir_water_thresh already set above

    
    # --- MASKS ---
    
    # 1. SHADOW MASK (The Problem Solver)
    # Shadows are DARK (< 1500)
    is_shadow = (brightness < shadow_bright_thresh)
    
    # 2. GLACIER MASK
    # Must be BRIGHT (> 2200). 
    # Must NOT be Vegetation (NDVI < 0.2).
    # Must NOT be Water (NDWI < 0.2).
    is_glacier = (brightness > bright_thresh) & (ndvi < 0.2) & (ndwi < 0.3)
    
    # 3. LAKE MASK
    # Water: High NDWI, Dark NIR.
    # But NOT Shadow (Shadows can have high NDWI sometimes).
    # Real water is darker in NIR than shadows usually, but shadows are dark in ALL bands.
    is_lake = (ndwi > 0.15) & (nir < nir_thresh) & (~is_shadow)
    
    # Clean up overlaps
    # Glaciers can't be shadows
    # Lakes can't be glaciers
    is_glacier = is_glacier & (~is_shadow) & (~is_lake)
    is_lake = is_lake & (~is_glacier)
    
    # Non-Glacier is everything else (Rocks + Shadows + Vegetation)
    is_nonglacier = (~is_glacier) & (~is_lake)
    
    # Filtering
    glacier_clean = median_filter(is_glacier.astype(np.uint8), size=3)
    lake_clean = median_filter(is_lake.astype(np.uint8), size=3)
    nonglacier_clean = median_filter(is_nonglacier.astype(np.uint8), size=3)
    
    return {
        "glacier": glacier_clean,
        "glacier_lake": lake_clean,
        "nonglacier": nonglacier_clean
    }, transform, crs

def process_universal(vrt_dir, json_out_dir, img_out_dir):
    os.makedirs(json_out_dir, exist_ok=True)
    os.makedirs(img_out_dir, exist_ok=True)
    
    vrt_files = sorted(glob.glob(os.path.join(vrt_dir, "*.vrt")))
    print(f"Applying Universal Shadow-Removing Logic to {len(vrt_files)} scenes...")
    
    # Setup Grid
    n_images = len(vrt_files)
    cols = 6
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
    fig.suptitle('Universal Consistent Classification (Shadows Removed)\nGrey: Glacier | Blue: Lake | Brown: Non-Glacier', fontsize=24, y=0.99)
    axes = axes.flatten()
    cmap = ListedColormap(['#8B4513', '#696969', '#00BFFF']) # Darker Grey for contrast

    for i, vrt_path in enumerate(vrt_files):
        date_str = os.path.basename(vrt_path)[10:20]
        print(f"[{i+1}/{n_images}] {date_str}...", end="\r")
        
        try:
            masks, transform, crs = classify_universal(vrt_path)
            
            # Save GeoJSON
            for class_name, mask_arr in masks.items():
                geoms = []
                # Simple optimization: only vectorize if enough pixels
                if np.sum(mask_arr) > 0:
                    for geom, val in shapes(mask_arr, mask=(mask_arr==1), transform=transform):
                        geoms.append(shape(geom))
                        
                if geoms:
                    gdf = gpd.GeoDataFrame({'geometry': geoms, 'date': date_str, 'class': class_name}, crs=crs)
                    gdf.to_file(os.path.join(json_out_dir, f"{class_name}_{date_str}.geojson"), driver='GeoJSON')
            
            # Plot
            display_img = np.zeros(masks['glacier'].shape, dtype=np.uint8) 
            # 0 = Brown (Non)
            display_img[masks['glacier'] == 1] = 1 # Grey
            display_img[masks['glacier_lake'] == 1] = 2 # Blue
            
            axes[i].imshow(display_img, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
            axes[i].set_title(date_str, fontsize=10)
            axes[i].axis('off')
            
        except Exception as e:
            print(f"Error {date_str}: {e}")

    # Clear empty
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    out_path = os.path.join(img_out_dir, "Final_Universal_Grid.png")
    print(f"\nSaving Universal Grid: {out_path}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

if __name__ == "__main__":
    vrt_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"
    json_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\universal_vectors"
    img_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\final_visuals"
    process_universal(vrt_dir, json_out, img_out)





