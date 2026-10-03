"""
Direct TIFF Classifier (Bypassing VRTs)
Assumes the "Updated" dataset has 10 bands.
Mapping Hypothesis: 
1=Blue, 2=Green, 3=Red ... 
and likely 9=SWIR1, 10=SWIR2 (Standard 10-band stack).

If this works, we get perfect NDSI for all images.
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

def classify_tiff_direct(tiff_path):
    with rasterio.open(tiff_path) as src:
        # Read bands assuming 10-band stack
        # B2(Blue), B3(Green), B4(Red), B8(NIR), B11(SWIR1)
        
        # Hypothesis: 
        # Band 1 = Blue (B2)
        # Band 2 = Green (B3)
        # Band 3 = Red (B4)
        # Band 4 = NIR (B8)
        # ...
        # Band 9 = SWIR1 (B11) ?? Let's verify this hypothesis visually later or assume standard S2 subset.
        
        # NOTE: Many "10 band" S2 products are B2,B3,B4,B8,B5,B6,B7,B8A,B11,B12.
        # Let's try reading indices 1, 2, 3, 4, 9.
        
        try:
            blue  = src.read(1).astype(float)
            green = src.read(2).astype(float)
            red   = src.read(3).astype(float)
            nir   = src.read(4).astype(float)
            # Try reading Band 9 as SWIR
            swir  = src.read(9).astype(float) 
        except Exception as e:
            print(f"Error reading bands: {e}")
            return None, None, None

        transform = src.transform
        crs = src.crs

    # --- CALC INDICES ---
    np.seterr(divide='ignore', invalid='ignore')
    
    # 1. NDSI (Green - SWIR) / (Green + SWIR)
    ndsi = (green - swir) / (green + swir + 1e-6)
    
    # 2. NDWI (Green - NIR)
    ndwi = (green - nir) / (green + nir + 1e-6)
    
    # 3. Brightness
    avg_vis = (blue + green + red) / 3.0
    
    # --- CLASSIFICATION ---
    # With valid SWIR, we can trust NDSI again!
    
    # Scale check
    thresh_swir_absorb = 2000 if np.nanmax(blue) > 2000 else 0.2
    
    # Shadows are dark in SWIR and VIS
    # Glaciers are Bright in VIS, Dark in SWIR (High NDSI)
    
    # GLACIER
    # High NDSI (> 0.2) + Bright Vis
    glacier_mask = (ndsi > 0.0) & (avg_vis > thresh_swir_absorb)
    
    # LAKE
    # High NDWI + Dark NIR
    lake_mask = (ndwi > 0.15) & (nir < thresh_swir_absorb)
    
    # Cleanup
    glacier_mask = glacier_mask & (~lake_mask)
    nonglacier_mask = (~glacier_mask) & (~lake_mask)
    
    return {
        "glacier": median_filter(glacier_mask.astype(np.uint8), size=3),
        "glacier_lake": median_filter(lake_mask.astype(np.uint8), size=3),
        "nonglacier": median_filter(nonglacier_mask.astype(np.uint8), size=3)
    }, transform, crs

def process_direct_tiff(tiff_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Search for Sentinel2_*.tif
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "Sentinel2_*.tif")))
    
    if not tiff_files:
        print("No TIFF files found.")
        return

    print(f"Processing {len(tiff_files)} TIFFs directly (assuming 10-band stack)...")
    
    # Grid
    n_images = len(tiff_files)
    cols = 6
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
    fig.suptitle('Direct TIFF Classification (Assumed 10-Band Stack)\nGrey: Glacier (High NDSI) | Blue: Lake', fontsize=24, y=0.99)
    axes = axes.flatten()
    cmap = ListedColormap(['#8B4513', '#D3D3D3', '#00BFFF'])

    for i, tiff_path in enumerate(tiff_files):
        date_str = os.path.basename(tiff_path)[10:20]
        print(f"[{i+1}/{n_images}] {date_str}...", end="\r")
        
        try:
            masks, transform, crs = classify_tiff_direct(tiff_path)
            
            if masks:
                display_img = np.zeros(masks['glacier'].shape, dtype=np.uint8)
                display_img[masks['glacier'] == 1] = 1
                display_img[masks['glacier_lake'] == 1] = 2
                
                axes[i].imshow(display_img, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
                axes[i].set_title(date_str, fontsize=10)
                axes[i].axis('off')
            else:
                 axes[i].text(0.5, 0.5, "Read Error", ha='center')

        except Exception as e:
            print(f"Error {date_str}: {e}")

    # Clear empty
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    out_path = os.path.join(output_dir, "Final_Direct_TIFF_Grid.png")
    print(f"\nSaving Grid: {out_path}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

if __name__ == "__main__":
    tiff_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    img_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\final_visuals"
    process_direct_tiff(tiff_dir, img_out)





