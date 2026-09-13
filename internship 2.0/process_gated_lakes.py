"""
Brightness-Gated Lake Classifier
Fixes "Shadows detected as Lakes" by enforcing a minimum brightness for water.
Logic:
1. Real Glacial Lakes are turquoise/milky -> Moderate Brightness.
2. Shadows are Pitch Black -> Low Brightness.
3. Rule: Lake = (NDWI > 0.15) AND (Brightness > Shadow_Cutoff).
This removes the blue noise in dark red-circle areas while keeping the green-circle lakes.
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
from scipy.ndimage import median_filter, label, sum as ndi_sum

def classify_brightness_gated(tiff_path):
    with rasterio.open(tiff_path) as src:
        try:
            # 10-band stack: 1=Blue, 2=Green, 3=Red, 4=NIR... 9=SWIR
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
    
    # Scale Normalization
    max_val = np.nanmax(blue)
    scale = 10000.0 if max_val > 2000 else 1.0
    
    # --- THRESHOLDS ---
    
    # 1. Shadow Cutoff (The Fix)
    # Shadows are usually < 1500 DN.
    # Lakes (sediment) are usually > 2000 DN.
    # We set a cautious cutoff around 1200-1500.
    shadow_cutoff = 0.12 * scale 
    
    # 2. LAKE Identification
    # Must have High NDWI
    # Must NOT be darker than the Shadow Cutoff
    # Must have low NIR (water strictness)
    is_lake = (ndwi > 0.15) & (avg_vis > shadow_cutoff) & (nir < (0.25 * scale))
    
    # 3. GLACIER Identification
    # Brightness > Higher cutoff
    glacier_cutoff = 0.25 * scale
    if ndsi is not None:
        is_glacier = (ndsi > 0.0) & (avg_vis > glacier_cutoff)
    else:
        ndvi = (nir - red) / (nir + red + 1e-6)
        is_glacier = (avg_vis > glacier_cutoff) & (ndvi < 0.2)
        
    # Overlap Resolution
    is_glacier = is_glacier & (~is_lake)
    
    # --- FILTERING ---
    # Remove tiny noise (1-10 pixels)
    def remove_speckles(mask, min_size=15):
        labeled, num = label(mask)
        if num == 0: return mask
        sizes = ndi_sum(mask, labeled, range(num+1))
        keep = sizes >= min_size
        return keep[labeled]

    is_lake = remove_speckles(is_lake, min_size=15)
    is_glacier = remove_speckles(is_glacier, min_size=15)
    
    is_nonglacier = (~is_glacier) & (~is_lake)
    
    return {
        "glacier": is_glacier.astype(np.uint8),
        "glacier_lake": is_lake.astype(np.uint8),
        "nonglacier": is_nonglacier.astype(np.uint8)
    }, transform, crs

def process_gated(tiff_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "Sentinel2_*.tif")))
    print(f"Processing {len(tiff_files)} scenes (Brightness Gated)...")
    
    n_images = len(tiff_files)
    cols = 6
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
    fig.suptitle('Brightness-Gated Classification\nRemoves Dark Shadow Noise from Lakes', fontsize=24, y=0.99)
    axes = axes.flatten()
    cmap = ListedColormap(['#8B4513', '#D3D3D3', '#00BFFF'])

    for i, tiff_path in enumerate(tiff_files):
        date_str = os.path.basename(tiff_path)[10:20]
        print(f"[{i+1}/{n_images}] {date_str}...", end="\r")
        
        try:
            masks, transform, crs = classify_brightness_gated(tiff_path)
            
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

    out_path = os.path.join(output_dir, "Final_Brightness_Gated_Grid.png")
    print(f"\nSaving Grid: {out_path}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

if __name__ == "__main__":
    tiff_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    img_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\final_visuals"
    process_gated(tiff_dir, img_out)



