"""
Cleaned Glacier Map (Area Filtered)
Method:
1. High-Quality Multi-Band Classification (swir/vis/nir).
2. "Speckle Removal": Filters out small polygons (< 10 pixels) for Lakes and Glaciers.
   - This removes the "unwanted lakes" (shadow noise).
   - Keeps the main glacier body intact.
3. Generates CLEAN GeoJSONs and High-Res Visual Grid.
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

def classify_and_clean(tiff_path):
    with rasterio.open(tiff_path) as src:
        # Read bands directly (10-band or 4-band support)
        try:
            blue  = src.read(1).astype(float)
            green = src.read(2).astype(float)
            red   = src.read(3).astype(float)
            nir   = src.read(4).astype(float)
            # Try getting SWIR (Band 9 in 10-band stack)
            swir = None
            if src.count >= 9:
                swir = src.read(9).astype(float)
        except:
            return None, None, None

        transform = src.transform
        crs = src.crs

    # --- INDICES ---
    np.seterr(divide='ignore', invalid='ignore')
    
    # NDWI (Water)
    ndwi = (green - nir) / (green + nir + 1e-6)
    
    # NDSI (Ice) - Prefer SWIR
    if swir is not None:
        ndsi = (green - swir) / (green + swir + 1e-6)
    else:
        ndsi = None
        
    # Brightness
    avg_vis = (blue + green + red) / 3.0
    
    # Scale Check
    scale = 10000.0 if np.nanmax(blue) > 2000 else 1.0
    
    # --- MASKS ---
    
    # 1. LAKE
    # Water: High NDWI + Dark NIR
    # Filter: Must not be super dark (deep shadow noise) unless NDWI is extremely high
    is_lake = (ndwi > 0.15) & (nir < (0.2 * scale))
    
    # 2. GLACIER
    if ndsi is not None:
        # High Quality: NDSI > 0, Bright
        is_glacier = (ndsi > 0.0) & (avg_vis > (0.25 * scale))
    else:
        # Fallback: Bright + Low Veg
        ndvi = (nir - red) / (nir + red + 1e-6)
        is_glacier = (avg_vis > (0.25 * scale)) & (ndvi < 0.2)
        
    # Cleanup Overlaps
    is_glacier = is_glacier & (~is_lake)
    
    # --- SPECKLE FILTER (The Fix) ---
    # We remove small "noise" lakes and glaciers
    
    def filter_noise(mask, min_size=15): # 15 pixels area threshold
        # Label connected components
        from scipy.ndimage import label, sum as ndi_sum
        labeled, num_features = label(mask)
        if num_features == 0: return mask
        
        sizes = ndi_sum(mask, labeled, range(num_features + 1))
        # Keep only large components
        mask_size = sizes < min_size
        remove_pixel = mask_size[labeled]
        mask[remove_pixel] = 0
        return mask
    
    # Filter Lakes (Removes shadow speckles)
    is_lake = filter_noise(is_lake, min_size=20) # Aggressive on lakes
    
    # Filter Glaciers (Removes snow patches on rocks)
    is_glacier = filter_noise(is_glacier, min_size=10)
    
    # Non-Glacier
    is_nonglacier = (~is_glacier) & (~is_lake)
    
    return {
        "glacier": is_glacier.astype(np.uint8),
        "glacier_lake": is_lake.astype(np.uint8),
        "nonglacier": is_nonglacier.astype(np.uint8)
    }, transform, crs

def process_clean(tiff_dir, json_out_dir, img_out_dir):
    os.makedirs(json_out_dir, exist_ok=True)
    os.makedirs(img_out_dir, exist_ok=True)
    
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "Sentinel2_*.tif")))
    print(f"Processing {len(tiff_files)} scenes with Area Filtering...")
    
    # Grid
    n_images = len(tiff_files)
    cols = 6
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
    fig.suptitle('Cleaned High-Quality Classification (Speckle Filtered)\nGrey: Glacier | Blue: Lake | Brown: Non-Glacier', fontsize=24, y=0.99)
    axes = axes.flatten()
    cmap = ListedColormap(['#8B4513', '#D3D3D3', '#00BFFF'])

    for i, tiff_path in enumerate(tiff_files):
        date_str = os.path.basename(tiff_path)[10:20]
        print(f"[{i+1}/{n_images}] {date_str}...", end="\r")
        
        try:
            masks, transform, crs = classify_and_clean(tiff_path)
            
            if masks:
                # Save GeoJSON
                for class_name, mask_arr in masks.items():
                    geoms = []
                    # Vectorize
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
            else:
                 axes[i].text(0.5, 0.5, "Error", ha='center')

        except Exception as e:
            print(f"Error {date_str}: {e}")

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    out_path = os.path.join(img_out_dir, "Final_Cleaned_Grid.png")
    print(f"\nSaving Cleaned Grid: {out_path}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

if __name__ == "__main__":
    tiff_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    json_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\glacier_cleaned_vectors"
    img_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\final_visuals"
    process_clean(tiff_dir, json_out, img_out)





