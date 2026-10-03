"""
Generate Classified Glacier GeoJSONs (Vector Format)
Constraints: Uses ONLY Bands 2, 3, 4, 8 (No SWIR/NDSI).
Method: Brightness & NIR-based thresholds.
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import glob
from scipy.ndimage import median_filter

def process_scene(vrt_path, output_dir):
    date_str = os.path.basename(vrt_path)[10:20] # Extract date '2021-11-08'
    print(f"Processing {date_str}...")

    with rasterio.open(vrt_path) as src:
        # User confirmed we have Bands 2, 3, 4, 8.
        # We need to map them correctly.
        # IF the VRT has 4 bands, we assume 1=B2, 2=B3, 3=B4, 4=B8.
        # IF the VRT has more, we try to grab the specific indices 2,3,4,8.
        
        nbands = src.count
        
        if nbands == 4:
            # Assumed Stack: Blue, Green, Red, NIR
            blue  = src.read(1).astype(float)
            green = src.read(2).astype(float)
            red   = src.read(3).astype(float)
            nir   = src.read(4).astype(float)
        else:
            # Standard Sentinel 2 mapping (1-based)
            # Try to read specific bands 2, 3, 4, 8 if they exist
            try:
                blue  = src.read(2).astype(float)
                green = src.read(3).astype(float)
                red   = src.read(4).astype(float)
                nir   = src.read(8).astype(float)
            except:
                # Fallback: maybe it's just RGBNir in 1,2,3,4 slots?
                blue  = src.read(1).astype(float)
                green = src.read(2).astype(float)
                red   = src.read(3).astype(float)
                nir   = src.read(4).astype(float)

        transform = src.transform
        crs = src.crs

    # Normalize values (0-1 range usually helps with thresholds if raw DNs vary)
    # Sentinel-2 raw is usually 0-10000.
    # We'll work with raw values but keep ratios which are scale-independent.
    
    np.seterr(divide='ignore', invalid='ignore')
    
    # INDICES (Using only B2, B3, B4, B8)
    
    # NDWI (Water) = (Green - NIR) / (Green + NIR)
    ndwi = (green - nir) / (green + nir + 1e-6)
    
    # NDVI (Veg) = (NIR - Red) / (NIR + Red)
    ndvi = (nir - red) / (nir + red + 1e-6)
    
    # Brightness (Visible average)
    brightness = (blue + green + red) / 3.0

    # CLASSIFICATION LOGIC (No SWIR)
    
    # Dynamic Brightness Thresholding
    # Snow/Ice is the brightest thing in the scene (besides clouds).
    # We find the 80th percentile to guess 'bright' pixels if simple threshold fails.
    # typically snow > 4000 DN.
    bright_thresh = 3000 # Conservative raw DN threshold for snow
    if np.nanmax(blue) < 1.0: # If data is 0-1 float
         bright_thresh = 0.3
         
    # Glacier Rule:
    # 1. Very Bright
    # 2. Not Vegetation (NDVI < 0.2)
    # 3. Not Water (NDWI < 0.2)
    glacier_mask = (brightness > bright_thresh) & (ndvi < 0.2) & (ndwi < 0.3)
    
    # Lake Rule:
    # 1. High NDWI (Water)
    # 2. Dark in NIR (Water absorbs NIR) - helps filter snow shadows
    # 3. Not super bright (filters bright snow that looks like water)
    lake_mask = (ndwi > 0.1) & (nir < bright_thresh * 0.5) 
    
    # Resolve Overlap: Lake beats Glacier (water on ice)
    glacier_mask = glacier_mask & (~lake_mask)
    
    # Non-Glacier: The rest
    nonglacier_mask = (~glacier_mask) & (~lake_mask)

    # Clean (Median Filter) to make it polygon-like
    glacier_clean = median_filter(glacier_mask.astype(np.uint8), size=3)
    lake_clean = median_filter(lake_mask.astype(np.uint8), size=3)
    nonglacier_clean = median_filter(nonglacier_mask.astype(np.uint8), size=3)

    classes = {
        "glacier": glacier_clean,
        "glacier_lake": lake_clean,
        "nonglacier": nonglacier_clean
    }

    # Vectorize and Save
    for class_name, mask_arr in classes.items():
        geoms = []
        for geom, val in shapes(mask_arr, mask=(mask_arr==1), transform=transform):
            geoms.append(shape(geom))
        
        if geoms:
            gdf = gpd.GeoDataFrame({'geometry': geoms}, crs=crs)
            # Add date attribute
            gdf['date'] = date_str
            gdf['class'] = class_name
            
            out_filename = f"{class_name}_{date_str}.geojson"
            out_path = os.path.join(output_dir, out_filename)
            gdf.to_file(out_path, driver='GeoJSON')

def process_all_geojson(vrt_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vrt_files = sorted(glob.glob(os.path.join(vrt_dir, "*.vrt")))
    
    if not vrt_files:
        print("No VRT files found.")
        return

    print(f"Found {len(vrt_files)} VRTs. Generating GeoJSON using Bands 2,3,4,8...")
    
    for i, f in enumerate(vrt_files):
        try:
            process_scene(f, output_dir)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print("Done! All GeoJSONs generated.")

if __name__ == "__main__":
    vrt_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"
    output_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\glacier_geojson_output"
    process_all_geojson(vrt_dir, output_dir)





