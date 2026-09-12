"""
Advanced Multi-Band Glacier Classifier
Uses NDSI (with SWIR) where available to perfectly distinguish Shadows from Glaciers.
Fallbacks to NIR/Brightness ratios for subsets with fewer bands.
Output: Cleaned Vector GeoJSONs + Visual Grid.
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

def get_bands(src):
    """Smartly fetch B2, B3, B4, B8, B11 based on availability"""
    nbands = src.count
    
    # Initialize with None
    b2, b3, b4, b8, b11 = None, None, None, None, None
    
    try:
        # Standard Sentinel-2 L1C/L2A mapping
        # 2:Blue, 3:Green, 4:Red, 8:NIR, 11:SWIR
        # Check against count to avoid index errors
        if nbands >= 2: b2 = src.read(2).astype(float)
        if nbands >= 3: b3 = src.read(3).astype(float)
        if nbands >= 4: b4 = src.read(4).astype(float)
        if nbands >= 8: b8 = src.read(8).astype(float)
        if nbands >= 11: b11 = src.read(11).astype(float)
        
        # Fallback for 4-band subset files (often just 1,2,3,4)
        if nbands == 4 and b8 is None:
            b2 = src.read(1).astype(float)
            b3 = src.read(2).astype(float)
            b4 = src.read(3).astype(float)
            b8 = src.read(4).astype(float)
            b11 = None # No SWIR in 4-band stack
            
    except Exception as e:
        print(f"Warning reading bands: {e}")
            
    return b2, b3, b4, b8, b11

def classify_scene_multiband(vrt_path):
    with rasterio.open(vrt_path) as src:
        blue, green, red, nir, swir = get_bands(src)
        transform = src.transform
        crs = src.crs

    if blue is None or green is None:
        return None, None, None # Failed read

    # --- INDICES CALCULATION ---
    np.seterr(divide='ignore', invalid='ignore')

    # 1. NDWI (Water) = (Green - NIR) / (Green + NIR)
    # Using Green(B3) and NIR(B8)
    ndwi = np.zeros_like(blue)
    if nir is not None:
        ndwi = (green - nir) / (green + nir + 1e-6)

    # 2. NDVI (Vegetation) = (NIR - Red) / (NIR + Red)
    ndvi = np.zeros_like(blue)
    if nir is not None and red is not None:
        ndvi = (nir - red) / (nir + red + 1e-6)
        
    # 3. NDSI (Snow/Ice) = (Green - SWIR) / (Green + SWIR)
    # **Critical for Shadow Removal**
    ndsi = None
    if swir is not None:
        ndsi = (green - swir) / (green + swir + 1e-6)

    # 4. Brightness / NIR Intensity
    # Shadows have extremely low NIR. Glaciers have high NIR.
    avg_vis = (blue + green + red) / 3.0
    
    # Auto-Thresholds (handle 0-1 vs 0-10000 scale)
    is_float_scale = np.nanmax(blue) < 100
    nir_thresh = 0.15 if is_float_scale else 1500
    vis_thresh = 0.25 if is_float_scale else 2500

    # --- CLASSIFICATION LOGIC ---
    
    # Init Masks
    glacier_mask = np.zeros_like(blue, dtype=bool)
    lake_mask = np.zeros_like(blue, dtype=bool)
    
    # LAKE DETECTION
    # High Water Index + Dark in NIR (Shadows are also dark, but have lower NDWI usually)
    # Use Blue band check: Lakes are blue/green. Shadows are 'black'.
    # Pure water absorbs NIR (val ~ 0).
    lake_mask = (ndwi > 0.15) & (nir < nir_thresh) & (blue > (nir * 1.5))

    # GLACIER DETECTION
    if ndsi is not None:
        # STRATEGY A: WE HAVE SWIR (The Best Way)
        # Snow: High NDSI (> 0.2), High Vis
        # Shadow: Low/Mod NDSI, LOW Vis
        # We enforce a brightness check to kill shadows that might sneak into NDSI
        glacier_mask = (ndsi > 0.0) & (avg_vis > vis_thresh) & (ndvi < 0.2)
    else:
        # STRATEGY B: NO SWIR (The Fallback)
        # Use NIR and Brightness strictly
        # Glaciers are the BRIGHTEST objects.
        # Shadows are the DARKEST.
        # We set a high threshold.
        glacier_mask = (avg_vis > vis_thresh) & (nir > nir_thresh) & (ndvi < 0.2)
    
    # Priority cleanup
    glacier_mask = glacier_mask & (~lake_mask) # Lake wins
    nonglacier_mask = (~glacier_mask) & (~lake_mask)

    # Filtering (Clean up noise)
    glacier_clean = median_filter(glacier_mask.astype(np.uint8), size=3)
    lake_clean = median_filter(lake_mask.astype(np.uint8), size=3)
    nonglacier_clean = median_filter(nonglacier_mask.astype(np.uint8), size=3)
    
    return {
        "glacier": glacier_clean,
        "glacier_lake": lake_clean,
        "nonglacier": nonglacier_clean
    }, transform, crs

def process_and_visualize(vrt_dir, json_out_dir, img_out_dir):
    os.makedirs(json_out_dir, exist_ok=True)
    os.makedirs(img_out_dir, exist_ok=True)
    
    vrt_files = sorted(glob.glob(os.path.join(vrt_dir, "*.vrt")))
    print(f"Processing {len(vrt_files)} scenes with Multi-Band Logic...")
    
    # Setup Grid Plot
    n_images = len(vrt_files)
    cols = 6
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
    fig.suptitle('Accurate Multi-Band Classification (Shadows Removed)\nGrey: Glacier | Blue: Lake | Brown: Non-Glacier', fontsize=24, y=0.99)
    axes = axes.flatten()
    cmap = ListedColormap(['#8B4513', '#B0B0B0', '#00BFFF']) # 0=Brown, 1=Grey, 2=Blue (We'll map indices to this)

    for i, vrt_path in enumerate(vrt_files):
        date_str = os.path.basename(vrt_path)[10:20]
        print(f"[{i+1}/{n_images}] {date_str}...", end="\r")
        
        try:
            masks, transform, crs = classify_scene_multiband(vrt_path)
            
            if masks:
                # 1. SAVE GEOJSONs
                for class_name, mask_arr in masks.items():
                    geoms = []
                    for geom, val in shapes(mask_arr, mask=(mask_arr==1), transform=transform):
                        geoms.append(shape(geom))
                    if geoms:
                        gdf = gpd.GeoDataFrame({'geometry': geoms, 'date': date_str, 'class': class_name}, crs=crs)
                        gdf.to_file(os.path.join(json_out_dir, f"{class_name}_{date_str}.geojson"), driver='GeoJSON')

                # 2. PREPARE DISPLAY IMAGE
                # Map masks to 0,1,2 for imshow
                # 0=Non-Glacier (Brown), 1=Glacier (Grey), 2=Lake (Blue)
                # Initialize with Brown
                display_img = np.zeros(masks['glacier'].shape, dtype=np.uint8) 
                display_img[masks['glacier'] == 1] = 1
                display_img[masks['glacier_lake'] == 1] = 2
                
                # Plot
                axes[i].imshow(display_img, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
                axes[i].set_title(date_str, fontsize=10)
                axes[i].axis('off')
            else:
                 axes[i].text(0.5, 0.5, "Read Error", ha='center')

        except Exception as e:
            print(f"Error {date_str}: {e}")
            axes[i].text(0.5, 0.5, "Error", ha='center')

    # Clear unused plots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    # Save Grid
    grid_path = os.path.join(img_out_dir, "Final_MultiBand_Grid.png")
    print(f"\nSaving Visual Grid to {grid_path}...")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=200)
    print("Done!")

if __name__ == "__main__":
    vrt_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"
    json_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\final_vectors_geojson"
    img_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\final_visuals"
    process_and_visualize(vrt_dir, json_out, img_out)



