"""
Pixel-Precise Classification (Shadow vs Lake Differentiation)
Method:
1. Uses 10-Band Stack (Bands 1,2,3,4,9) for max info.
2. Differentiates Lake vs Shadow using strict NIR absorbance:
   - Lakes: High NDWI + Extremely LOW NIR (< 1500).
   - Shadows: Moderate NDWI + Higher NIR (> 1500).
3. Recover Lakes: Prioritizes Lake detection over Shadow detection if water signal is strong.
4. Output: GeoJSON Vectors + Visual Grid.
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

def classify_precision(tiff_path):
    with rasterio.open(tiff_path) as src:
        try:
            # Band Hypothesis: 10-band stack
            # 1=Blue, 2=Green, 3=Red, 4=NIR, 9=SWIR
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
    
    # Scale Check
    max_val = np.nanmax(blue)
    scale = 10000.0 if max_val > 2000 else 1.0
    
    # --- PIXEL LOGIC ---
    
    # 1. GLACIER (Bright + High NDSI)
    # Strict brightness to avoid "dark ice" which is often shadow
    glacier_thresh = 0.3 * scale
    if ndsi is not None:
        is_glacier = (ndsi > 0.0) & (avg_vis > glacier_thresh)
    else:
        # Fallback
        ndvi = (nir - red) / (nir + red + 1e-6)
        is_glacier = (avg_vis > glacier_thresh) & (ndvi < 0.2)
        
    # 2. LAKE (High NDWI + Dark NIR)
    # The Key: Water absorbs NIR. Shadows don't (as much).
    nir_water_limit = 0.18 * scale # Strict!
    is_lake = (ndwi > 0.15) & (nir < nir_water_limit)
    
    # 3. SHADOW (The cleanup)
    # Anything else dark is a shadow (Non-Glacier)
    # We don't perform a specific "Shadow Mask", we just rely on Lake being strict.
    
    # Overlap Resolution
    is_glacier = is_glacier & (~is_lake) # Lake wins (melt ponds)
    
    # --- SPECKLE FILTER ---
    def filter_noise(mask, min_size):
        labeled, num = label(mask)
        if num == 0: return mask
        sizes = ndi_sum(mask, labeled, range(num+1))
        # Keep features >= min_size
        keep = sizes >= min_size
        return keep[labeled]
    
    # Remove tiny noise lakes (Shadow artifacts)
    is_lake = filter_noise(is_lake, min_size=10)
    
    # Remove tiny glacier specs
    is_glacier = filter_noise(is_glacier, min_size=10)
    
    is_nonglacier = (~is_glacier) & (~is_lake)
    
    return {
        "glacier": is_glacier.astype(np.uint8),
        "glacier_lake": is_lake.astype(np.uint8),
        "nonglacier": is_nonglacier.astype(np.uint8)
    }, transform, crs

def process_precision(tiff_dir, json_out_dir, img_out_dir):
    os.makedirs(json_out_dir, exist_ok=True)
    os.makedirs(img_out_dir, exist_ok=True)
    
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "Sentinel2_*.tif")))
    print(f"Processing {len(tiff_files)} scenes (Precision Mode)...")
    
    # Visualization setup
    n_images = len(tiff_files)
    cols = 6
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
    fig.suptitle('Pixel-Precise Lake/Shadow Separation\nGrey: Glacier | Blue: Lake', fontsize=24, y=0.99)
    axes = axes.flatten()
    cmap = ListedColormap(['#8B4513', '#D3D3D3', '#00BFFF'])

    for i, tiff_path in enumerate(tiff_files):
        date_str = os.path.basename(tiff_path)[10:20]
        print(f"[{i+1}/{n_images}] {date_str}...", end="\r")
        
        try:
            masks, transform, crs = classify_precision(tiff_path)
            
            if masks:
                # 1. Save Vectors
                for class_name, mask_arr in masks.items():
                    geoms = []
                    # Vectorize
                    if np.sum(mask_arr) > 0:
                        for geom, val in shapes(mask_arr, mask=(mask_arr==1), transform=transform):
                            geoms.append(shape(geom))
                    if geoms:
                        gdf = gpd.GeoDataFrame({'geometry': geoms, 'date': date_str, 'class': class_name}, crs=crs)
                        gdf.to_file(os.path.join(json_out_dir, f"{class_name}_{date_str}.geojson"), driver='GeoJSON')
                
                # 2. Plot
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

    out_path = os.path.join(img_out_dir, "Final_Precision_Grid.png")
    print(f"\nSaving Precision Grid: {out_path}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

if __name__ == "__main__":
    tiff_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    json_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\final_precision_vectors"
    img_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\final_visuals"
    process_precision(tiff_dir, json_out, img_out)



