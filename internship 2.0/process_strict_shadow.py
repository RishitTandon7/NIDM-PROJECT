"""
Strict Shadow-Removal Classifier
Method: "Dark Filter" - Hard thresholding to force all shadows to Non-Glacier.
Logic:
1. Shadows: (Blue+Green+Red+NIR) < Threshold --> Non-Glacier (Brown)
2. Water: NDWI > Textures & Brightness in Blue --> Lake (Blue)
3. Glacier: Everything else that is Bright --> Glacier (Grey)
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

def classify_strict_shadow(tiff_path):
    with rasterio.open(tiff_path) as src:
        # Read B2, B3, B4, B8 + B9(SWIR)
        try:
            blue  = src.read(1).astype(float) # B2
            green = src.read(2).astype(float) # B3
            red   = src.read(3).astype(float) # B4
            nir   = src.read(4).astype(float) # B8
            swir  = src.read(9).astype(float) # B11 (Band 9 idx)
        except:
             # Fallback if 10-band read fails (e.g. only 4 bands)
            blue  = src.read(1).astype(float)
            green = src.read(2).astype(float)
            red   = src.read(3).astype(float)
            nir   = src.read(4).astype(float)
            swir = None

        transform = src.transform
        crs = src.crs

    # --- INDICES ---
    np.seterr(divide='ignore', invalid='ignore')
    
    # Sum of Vis/NIR (Total Brightness)
    total_bright = blue + green + red + nir
    
    # NDWI (Water)
    ndwi = (green - nir) / (green + nir + 1e-6)
    
    # NDSI (Ice) - Only if SWIR exists
    ndsi = None
    if swir is not None:
        ndsi = (green - swir) / (green + swir + 1e-6)

    # --- STRICT THRESHOLDS ---
    
    # 1. SHADOW KILLER
    # Shadows are dark in ALL bands.
    # Determine scale (0-1 or 0-10000)
    max_val = np.nanmax(blue)
    if max_val > 2000:
        shadow_thresh = 4000 # (Sum of 4 bands ~ average 1000 each)
        swir_dark_thresh = 1500
    else:
        shadow_thresh = 0.4
        swir_dark_thresh = 0.15

    # Mask: Dark Pixels -> Non-Glacier
    is_shadow = total_bright < shadow_thresh
    
    # 2. LAKE DETECTOR
    # Water is dark in NIR/SWIR but bright in Green/Blue.
    # NDWI > 0.15 usually.
    # Must NOT be shadow (Shadows is dark everywhere)
    is_lake = (ndwi > 0.15) & (~is_shadow)
    
    # 3. GLACIER DETECTOR
    if ndsi is not None:
        # Standard: High NDSI (>0.0), Brightness high enough
        is_glacier = (ndsi > 0.0) & (~is_shadow) & (~is_lake)
    else:
        # Fallback: Just Brightness + Low Vegetation
        ndvi = (nir - red) / (nir + red + 1e-6)
        is_glacier = (~is_shadow) & (~is_lake) & (ndvi < 0.2)
        
    # Cleanup
    is_glacier = median_filter(is_glacier.astype(np.uint8), size=3)
    is_lake = median_filter(is_lake.astype(np.uint8), size=3)
    is_nonglacier = median_filter((~is_glacier) & (~is_lake), size=3)
    
    return {
        "glacier": is_glacier,
        "glacier_lake": is_lake,
        "nonglacier": is_nonglacier
    }, transform, crs

def process_strict(tiff_dir, img_out_dir):
    os.makedirs(img_out_dir, exist_ok=True)
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "Sentinel2_*.tif")))
    
    # Define Segments as requested:
    # 20 images (4 cols & 5 rows) * 3 segments
    # 6 images * 1 segment
    segments = [
        {"name": "Segment_1", "start": 0, "count": 20, "cols": 4, "rows": 5},
        {"name": "Segment_2", "start": 20, "count": 20, "cols": 4, "rows": 5},
        {"name": "Segment_3", "start": 40, "count": 20, "cols": 4, "rows": 5},
        {"name": "Segment_4", "start": 60, "count": 6, "cols": 3, "rows": 2}, # 3x2 grid for 6 images
    ]
    
    print(f"Applying Strict Shadow Filter to {len(tiff_files)} scenes in {len(segments)} segments...")
    
    cmap = ListedColormap(['#8B4513', '#D3D3D3', '#00BFFF']) # Brown, Grey, Blue

    for seg in segments:
        start = seg["start"]
        end = start + seg["count"]
        # Ensure we don't exceed available files
        batch_files = tiff_files[start:end]
        
        if not batch_files:
            continue
            
        print(f"\nProcessing {seg['name']} ({len(batch_files)} images)...")
        
        cols = seg["cols"]
        rows = seg["rows"]
        
        # Calculate appropriate figure size
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        fig.suptitle(f'Strict Shadow-Removal - {seg["name"]}\nGrey: Glacier | Blue: Lake | Brown: Non-Glacier', fontsize=20, y=0.99)
        
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
            
        processed_count = 0
        for i, tiff_path in enumerate(batch_files):
            date_str = os.path.basename(tiff_path)[10:20]
            print(f"[{i+1}/{len(batch_files)}] {date_str}...", end="\r")
            
            try:
                masks, transform, crs = classify_strict_shadow(tiff_path)
                
                display_img = np.zeros(masks['glacier'].shape, dtype=np.uint8) 
                display_img[masks['glacier'] == 1] = 1 
                display_img[masks['glacier_lake'] == 1] = 2 
                
                axes[i].imshow(display_img, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
                axes[i].set_title(date_str, fontsize=10)
                axes[i].axis('off')
                processed_count += 1
                
            except Exception as e:
                print(f"Error {date_str}: {e}")
                axes[i].axis('off')

        # Hide unused axes if any
        for j in range(processed_count, len(axes)):
            axes[j].axis('off')

        out_path = os.path.join(img_out_dir, f"Final_Strict_Shadow_{seg['name']}.png")
        print(f"\nSaving {seg['name']}: {out_path}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

if __name__ == "__main__":
    tiff_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    img_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\segmented_outputs"
    process_strict(tiff_dir, img_out)





