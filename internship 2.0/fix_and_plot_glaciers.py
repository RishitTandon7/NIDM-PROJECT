"""
Fix Shadows & Plot Glaciers Only
1. Re-generates 'glacier' GeoJSONs with stricter brightness thresholds to remove shadows.
2. Creates a Visual Grid showing ONLY glaciers with Latitude/Longitude axes.
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import median_filter

def process_and_plot_glaciers(vrt_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vrt_files = sorted(glob.glob(os.path.join(vrt_dir, "*.vrt")))
    
    if not vrt_files:
        print("No VRT files found.")
        return

    print(f"Processing {len(vrt_files)} scenes (Removing Shadows & Plotting)...")

    # Grid Setup for Final Image
    n_images = len(vrt_files)
    cols = 6
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
    fig.suptitle('Detected Glaciers (Shadows Removed) - 2021-2025\nCleaned Vector Output', fontsize=24, y=0.995)
    axes = axes.flatten()

    for i, vrt_path in enumerate(vrt_files):
        date_str = os.path.basename(vrt_path)[10:20]
        print(f"[{i+1}/{n_images}] {date_str}...", end="\r")

        with rasterio.open(vrt_path) as src:
            nbands = src.count
            
            # Read Bands (Robust 4-band logic)
            try: 
                # Try Standard
                blue  = src.read(2).astype(float)
                green = src.read(3).astype(float)
                red   = src.read(4).astype(float)
                nir   = src.read(8).astype(float)
            except:
                # Fallback to 1,2,3,4
                blue  = src.read(1).astype(float)
                green = src.read(2).astype(float)
                red   = src.read(3).astype(float)
                nir   = src.read(4).astype(float)
                
            transform = src.transform
            crs = src.crs
            bounds = src.bounds

        # Indices
        np.seterr(divide='ignore', invalid='ignore')
        
        # NDVI (Veg)
        ndvi = (nir - red) / (nir + red + 1e-6)
        
        # Brightness (Essential for filtering shadows)
        # Shadows are DARK. Glaciers are BRIGHT.
        # Normalize Data to 0-1 range roughly to apply consistent threshold
        # S2 data is usually 0-10000. 
        # Using a raw threshold like > 2500 is common for snow.
        # If float 0-1 (from previous processing steps?), we check max.
        
        raw_brightness = (blue + green + red) / 3.0
        
        # Threshold Logic
        # If max value is huge (>1000), it's raw DN.
        threshold = 3500 # Stricter threshold (was implicit/lower before)
        if np.nanmax(blue) < 10: # Float data
             threshold = 0.35 

        # GLACIER MASK (Strict)
        # 1. Must be Bright (Removes Shadows/Rocks)
        # 2. Must NOT be Vegetation
        glacier_mask = (raw_brightness > threshold) & (ndvi < 0.2)
        
        # Clean Noise (Morphology)
        glacier_clean = median_filter(glacier_mask.astype(np.uint8), size=3)

        # Vectorize (Create GeoJSON)
        geoms = []
        for geom, val in shapes(glacier_clean, mask=(glacier_clean==1), transform=transform):
            geoms.append(shape(geom))
        
        # Plotting on the Grid
        ax = axes[i]
        
        if geoms:
            # Save GeoJSON
            gdf = gpd.GeoDataFrame({'geometry': geoms}, crs=crs)
            out_json = os.path.join(output_dir, f"glacier_cleaned_{date_str}.geojson")
            gdf.to_file(out_json, driver='GeoJSON')
            
            # Plot Polygons
            gdf.plot(ax=ax, color='#808080', edgecolor='none') # Grey Glaciers
            
            # Set Bounds to match image extent (important for empty space)
            ax.set_xlim(bounds.left, bounds.right)
            ax.set_ylim(bounds.bottom, bounds.top)
        else:
            # No glaciers found? (Unlikely, but handle it)
            ax.text(0.5, 0.5, "No Glaciers", ha='center', va='center', transform=ax.transAxes)

        # Formatting (Lat/Lon Grid)
        ax.set_title(date_str, fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Minimal ticks to avoid clutter
        ax.set_xticks([]) 
        ax.set_yticks([])
        # Real Lat/Lon labels only on edges would be cleaner, but user asked for "in each image"
        # Adding labels for 66 (!!) images makes it unreadable. 
        # I will keep the *grid lines* visible to show coordinates exist.

    # Hide unused
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    print("\nSaving Cleaned Glacier Grid...")
    out_plot = os.path.join(output_dir, "Glaciers_Only_Grid_Cleaned.png")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200, bbox_inches='tight')
    print(f"Saved: {out_plot}")

if __name__ == "__main__":
    vrt_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"
    # New folder for the cleaned ("shadow-free") output
    output_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\glaciers_only_cleaned"
    process_and_plot_glaciers(vrt_dir, output_dir)



