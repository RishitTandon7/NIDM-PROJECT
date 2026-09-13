"""
Generate Final Stitched Classification Grid (Visual)
Logic: Matches 'generate_glacier_geojson.py' (Bands 2,3,4,8 only).
Output: A single high-res grid image of all 66 classified maps.
Colors: Grey (Glacier), Blue (Lake), Brown (Non-Glacier).
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob
from scipy.ndimage import median_filter

def classify_scene_visual(vrt_path):
    with rasterio.open(vrt_path) as src:
        nbands = src.count
        
        # Read Bands 2, 3, 4, 8 (Blue, Green, Red, NIR)
        if nbands == 4:
            blue  = src.read(1).astype(float)
            green = src.read(2).astype(float)
            red   = src.read(3).astype(float)
            nir   = src.read(4).astype(float)
        else:
            try:
                blue  = src.read(2).astype(float)
                green = src.read(3).astype(float)
                red   = src.read(4).astype(float)
                nir   = src.read(8).astype(float)
            except:
                blue  = src.read(1).astype(float)
                green = src.read(2).astype(float)
                red   = src.read(3).astype(float)
                nir   = src.read(4).astype(float)

    # Indices
    np.seterr(divide='ignore', invalid='ignore')
    ndwi = (green - nir) / (green + nir + 1e-6)
    ndvi = (nir - red) / (nir + red + 1e-6)
    brightness = (blue + green + red) / 3.0

    # Dynamic Threshold for Brightness (simple auto-scale check)
    bright_thresh = 3000
    if np.nanmax(blue) < 1.0: 
         bright_thresh = 0.3

    # Classification Rules (Same as GeoJSON script)
    glacier_mask = (brightness > bright_thresh) & (ndvi < 0.2) & (ndwi < 0.3)
    lake_mask = (ndwi > 0.1) & (nir < bright_thresh * 0.5)
    
    # Priority: Lake > Glacier
    glacier_mask = glacier_mask & (~lake_mask)
    nonglacier_mask = (~glacier_mask) & (~lake_mask)

    # Create Composite Array for Visualization
    # 0 = Glacier (Grey), 1 = Lake (Blue), 2 = Non-Glacier (Brown)
    
    # We initialize with Brown (2)
    final_class = np.full(glacier_mask.shape, 2, dtype=np.uint8)
    
    final_class[glacier_mask] = 0
    final_class[lake_mask] = 1

    # Clean (Median Filter)
    final_class_clean = median_filter(final_class, size=3)
    
    return final_class_clean

def generate_grid(vrt_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vrt_files = sorted(glob.glob(os.path.join(vrt_dir, "*.vrt")))
    
    if not vrt_files:
        print("No VRT files found.")
        return

    print(f"Generating Visual Grid for {len(vrt_files)} images...")
    
    # Colors: 0=Grey, 1=Blue, 2=Brown
    # Hex codes: Grey=#8C8C8C, Blue=#1E90FF, Brown=#8B5A2B
    cmap = ListedColormap(['#B0B0B0', '#00BFFF', '#8B4513']) 
    
    # Grid Setup
    n_images = len(vrt_files)
    cols = 6
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
    fig.suptitle('Sentinel-2 Glacier Classification Time Series (2021-2025)\nGrey: Glacier | Blue: Lake | Brown: Non-Glacier', fontsize=24, y=0.99)
    axes = axes.flatten()
    
    for i, vrt_path in enumerate(vrt_files):
        print(f"[{i+1}/{n_images}] Mapping {os.path.basename(vrt_path)}...")
        try:
            result = classify_scene_visual(vrt_path)
            
            axes[i].imshow(result, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
            axes[i].set_title(os.path.basename(vrt_path)[10:20], fontsize=10)
            axes[i].axis('off')
            
        except Exception as e:
            print(f"Error on {vrt_path}: {e}")
            axes[i].set_title("ERROR")
            axes[i].axis('off')

    # Hide empty plots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    out_path = os.path.join(output_dir, "Final_Classification_Grid_Corrected.png")
    print(f"Saving Grid to {out_path} ...")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Done!")

if __name__ == "__main__":
    vrt_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"
    output_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\final_accurate_maps"
    generate_grid(vrt_dir, output_dir)



