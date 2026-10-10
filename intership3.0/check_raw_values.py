
import rasterio
import numpy as np
import glob
import os

tiff_dir = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1"
tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "S1_*.tif")))

if tiff_files:
    test_file = tiff_files[0]
    print(f"Checking file: {test_file}")
    
    with rasterio.open(test_file) as src:
        vv = src.read(1)
        vh = src.read(2)
        
        print("\n--- RAW DATA STATISTICS (No Conversion) ---")
        
        # VV
        print(f"VV - Min: {np.nanmin(vv):.4f}")
        print(f"VV - Max: {np.nanmax(vv):.4f}")
        print(f"VV - Mean: {np.nanmean(vv):.4f}")
        print(f"VV - Median: {np.nanmedian(vv):.4f}")
        p5, p95 = np.nanpercentile(vv, [5, 95])
        print(f"VV - 5th Percentile: {p5:.4f}")
        print(f"VV - 95th Percentile: {p95:.4f}")
        
        print("-" * 30)
        
        # VH
        print(f"VH - Min: {np.nanmin(vh):.4f}")
        print(f"VH - Max: {np.nanmax(vh):.4f}")
        print(f"VH - Mean: {np.nanmean(vh):.4f}")
        print(f"VH - Median: {np.nanmedian(vh):.4f}")
        p5, p95 = np.nanpercentile(vh, [5, 95])
        print(f"VH - 5th Percentile: {p5:.4f}")
        print(f"VH - 95th Percentile: {p95:.4f}")

else:
    print("No TIFF files found.")
