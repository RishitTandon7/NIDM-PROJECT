"""
Final Corrected Classification
Based on pixel-level analysis:
- Lakes: NDWI > 0.1 AND NIR < 4000 (relaxed from 1500)
- Shadows: Low brightness AND NDWI < 0.1
- Glaciers: High brightness
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

def classify_final(tiff_path):
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
    
    # --- FINAL CALIBRATED THRESHOLDS ---
    
    # 1. LAKE DETECTION (Relaxed NIR threshold)
    # Data shows: Lake has NDWI > 0.1 and NIR ~3000-4000
    # Shadows have NDWI < 0.1 OR NIR > 4000
    is_lake = (ndwi > 0.1) & (nir < 4500)
    
    # 2. SHADOW FILTER
    # Shadows are dark (avg_vis < 3000) AND don't have water signature
    is_shadow = (avg_vis < 3000) & (ndwi < 0.08)
    
    # Remove shadow false positives from lakes
    is_lake = is_lake & (~is_shadow)
    
    # 3. GLACIER DETECTION
    # Bright areas (avg_vis > 4000)
    if ndsi is not None:
        is_glacier = (ndsi > 0.0) & (avg_vis > 4000)
    else:
        ndvi = (nir - red) / (nir + red + 1e-6)
        is_glacier = (avg_vis > 4000) & (ndvi < 0.2)
        
    # Cleanup overlaps
    # Lake wins over glacier (supraglacial lakes)
    is_glacier = is_glacier & (~is_lake)
    
    is_nonglacier = (~is_glacier) & (~is_lake)
    
    # --- SPECKLE FILTER ---
    def remove_speckles(mask, min_size=10):
        labeled, num = label(mask)
        if num == 0: return mask
        sizes = ndi_sum(mask, labeled, range(num+1))
        keep = sizes >= min_size
        return keep[labeled]

    is_lake = remove_speckles(is_lake, min_size=8)
    is_glacier = remove_speckles(is_glacier, min_size=10)
    
    return {
        "glacier": is_glacier.astype(np.uint8),
        "glacier_lake": is_lake.astype(np.uint8),
        "nonglacier": is_nonglacier.astype(np.uint8)
    }, transform, crs

def process_final(tiff_dir, json_out_dir, output_dir):
    os.makedirs(json_out_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "Sentinel2_*.tif")))
    print(f"Processing {len(tiff_files)} scenes (FINAL Calibration)...")
    
    n_images = len(tiff_files)
    cols = 6
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
    fig.suptitle('FINAL Classification (Pixel-Calibrated)\nGrey: Glacier | Blue: Lake | Brown: Non-Glacier', fontsize=24, y=0.99)
    axes = axes.flatten()
    cmap = ListedColormap(['#8B4513', '#D3D3D3', '#00BFFF'])

    for i, tiff_path in enumerate(tiff_files):
        date_str = os.path.basename(tiff_path)[10:20]
        print(f"[{i+1}/{n_images}] {date_str}...", end="\r")
        
        try:
            masks, transform, crs = classify_final(tiff_path)
            
            if masks:
                # Save GeoJSON vectors
                for class_name, mask_arr in masks.items():
                    geoms = []
                    if np.sum(mask_arr) > 0:
                        for geom, val in shapes(mask_arr, mask=(mask_arr==1), transform=transform):
                            geoms.append(shape(geom))
                    if geoms:
                        gdf = gpd.GeoDataFrame({'geometry': geoms, 'date': date_str, 'class': class_name}, crs=crs)
                        gdf.to_file(os.path.join(json_out_dir, f"{class_name}_{date_str}.geojson"), driver='GeoJSON')
                
                # Plot
                display_img = np.zeros(masks['glacier'].shape, dtype=np.uint8)
                display_img[masks['glacier'] == 1] = 1
                display_img[masks['glacier_lake'] == 1] = 2
                
                axes[i].imshow(display_img, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
                axes[i].set_title(date_str, fontsize=10)
                axes[i].axis('off')
        except Exception as e:
            print(f"Error {date_str}: {e}")

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    out_path = os.path.join(output_dir, "FINAL_Classification_Grid.png")
    print(f"\nSaving FINAL Grid: {out_path}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

if __name__ == "__main__":
    tiff_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    json_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\FINAL_vectors"
    img_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\final_visuals"
    process_final(tiff_dir, json_out, img_out)





