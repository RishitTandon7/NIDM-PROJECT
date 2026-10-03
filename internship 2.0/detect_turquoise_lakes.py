"""
Turquoise Lake Detector
Glacial Lakes are TURQUOISE (high Green reflectance due to rock flour).
Shadows are uniformly DARK.
We detect lakes by their GREEN DOMINANCE, not just NDWI.

Logic:
1. Lake = (Green > Blue) AND (Green > Red) AND (NDWI > 0)
   This catches the turquoise color.
2. Shadow = Dark everywhere (Total < Threshold).
3. Glacier = Bright + NDSI > 0 (or brightness fallback).
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
from scipy.ndimage import label, sum as ndi_sum

def classify_turquoise_lakes(tiff_path):
    with rasterio.open(tiff_path) as src:
        try:
            blue  = src.read(1).astype(float)
            green = src.read(2).astype(float)
            red   = src.read(3).astype(float)
            nir   = src.read(4).astype(float)
            swir = None
            if src.count >= 9:
                swir = src.read(9).astype(float)
        except:
             return None, None, None
        
        transform = src.transform
        crs = src.crs

    np.seterr(divide='ignore', invalid='ignore')
    
    # --- INDICES ---
    ndwi = (green - nir) / (green + nir + 1e-6)
    
    ndsi = None
    if swir is not None:
        ndsi = (green - swir) / (green + swir + 1e-6)
    
    avg_vis = (blue + green + red) / 3.0
    total_bright = blue + green + red + nir
    
    # Scale
    max_val = np.nanmax(blue)
    scale = 10000.0 if max_val > 2000 else 1.0
    
    # --- TURQUOISE LAKE DETECTION ---
    # Turquoise = Green dominant
    # We check if Green is the highest of the visible bands
    # AND NIR is low (water absorbs NIR)
    
    is_green_dominant = (green > blue * 0.95) & (green > red * 1.0) # Green >= Blue and > Red
    is_nir_absorbing = nir < (0.3 * scale) # Water absorbs NIR
    is_not_too_dark = avg_vis > (0.08 * scale) # Filter pitch black shadows
    
    is_lake = is_green_dominant & is_nir_absorbing & is_not_too_dark
    
    # --- SHADOW DETECTION ---
    # Shadows are uniformly dark. No single band dominance.
    # Total brightness < Threshold
    shadow_thresh = 0.3 * scale * 4 # Sum of 4 bands
    is_shadow = total_bright < shadow_thresh
    
    # Shadows cannot be lakes
    is_lake = is_lake & (~is_shadow)
    
    # --- GLACIER DETECTION ---
    glacier_bright_thresh = 0.28 * scale
    if ndsi is not None:
        is_glacier = (ndsi > 0.0) & (avg_vis > glacier_bright_thresh)
    else:
        ndvi = (nir - red) / (nir + red + 1e-6)
        is_glacier = (avg_vis > glacier_bright_thresh) & (ndvi < 0.2)
        
    # Glacier cannot be lake or shadow
    is_glacier = is_glacier & (~is_lake) & (~is_shadow)
    
    # Non-Glacier = Everything else
    is_nonglacier = (~is_glacier) & (~is_lake)
    
    # --- SPECKLE FILTER ---
    def remove_speckles(mask, min_size=10):
        labeled, num = label(mask)
        if num == 0: return mask
        sizes = ndi_sum(mask, labeled, range(num+1))
        keep = sizes >= min_size
        return keep[labeled]

    is_lake = remove_speckles(is_lake, min_size=10)
    is_glacier = remove_speckles(is_glacier, min_size=10)
    
    return {
        "glacier": is_glacier.astype(np.uint8),
        "glacier_lake": is_lake.astype(np.uint8),
        "nonglacier": is_nonglacier.astype(np.uint8)
    }, transform, crs

def process_turquoise(tiff_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "Sentinel2_*.tif")))
    print(f"Processing {len(tiff_files)} scenes (Turquoise Lake Detection)...")
    
    n_images = len(tiff_files)
    cols = 6
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
    fig.suptitle('Turquoise Lake Detection (Green Dominance)\nGrey: Glacier | Blue: Lake | Brown: Non-Glacier', fontsize=24, y=0.99)
    axes = axes.flatten()
    cmap = ListedColormap(['#8B4513', '#D3D3D3', '#00CED1']) # Brown, Grey, Dark Cyan (turquoise)

    for i, tiff_path in enumerate(tiff_files):
        date_str = os.path.basename(tiff_path)[10:20]
        print(f"[{i+1}/{n_images}] {date_str}...", end="\r")
        
        try:
            masks, transform, crs = classify_turquoise_lakes(tiff_path)
            
            if masks:
                display_img = np.zeros(masks['glacier'].shape, dtype=np.uint8)
                display_img[masks['glacier'] == 1] = 1
                display_img[masks['glacier_lake'] == 1] = 2
                
                axes[i].imshow(display_img, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
                axes[i].set_title(date_str, fontsize=10)
                axes[i].axis('off')
        except:
            pass

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    out_path = os.path.join(output_dir, "Final_Turquoise_Lake_Grid.png")
    print(f"\nSaving Grid: {out_path}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

if __name__ == "__main__":
    tiff_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    img_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\final_visuals"
    process_turquoise(tiff_dir, img_out)





