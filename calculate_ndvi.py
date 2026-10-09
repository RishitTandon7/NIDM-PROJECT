import rasterio
import numpy as np
import sys
import os

def calculate_ndvi(input_tiff, output_tiff, red_band_idx, nir_band_idx):
    """
    Calculates NDVI from a multi-band TIFF image.
    
    Formula: NDVI = (NIR - Red) / (NIR + Red)
    """
    print(f"Loading image: {input_tiff}...")
    
    try:
        with rasterio.open(input_tiff) as src:
            # Read the Red and NIR bands. Rasterio bands are 1-indexed.
            print(f"Reading Red Band (Band {red_band_idx})...")
            red = src.read(red_band_idx).astype(np.float32)
            
            print(f"Reading NIR Band (Band {nir_band_idx})...")
            nir = src.read(nir_band_idx).astype(np.float32)
            
            # Prevent division by zero
            # We add a tiny value (epsilon) to the denominator
            epsilon = 1e-10
            
            print("Calculating NDVI array...")
            ndvi = (nir - red) / (nir + red + epsilon)
            
            # NDVI values technically range from -1 to 1.
            # We can optionally clip it just to be safe from anomalies
            ndvi = np.clip(ndvi, -1.0, 1.0)
            
            # Prepare metadata for the output file
            profile = src.profile
            profile.update(
                dtype=rasterio.float32,
                count=1,
                compress='lzw'
            )
            
            print(f"Saving NDVI to: {output_tiff}")
            with rasterio.open(output_tiff, 'w', **profile) as dst:
                dst.write(ndvi, 1)
                
        print("Success! NDVI calculation complete.")
        
    except Exception as e:
        print(f"Error calculating NDVI: {e}")

if __name__ == "__main__":
    # Example usage. You can change these paths to your actual optical .tif files!
    # For Sentinel-2: Red = Band 4, NIR = Band 8
    # For Landsat 8/9: Red = Band 4, NIR = Band 5
    
    if len(sys.argv) == 5:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        red_idx = int(sys.argv[3])
        nir_idx = int(sys.argv[4])
        calculate_ndvi(input_path, output_path, red_idx, nir_idx)
    else:
        print("Usage: python calculate_ndvi.py <input_tiff> <output_tiff> <red_band_number> <nir_band_number>")
        print("Example for Sentinel-2: python calculate_ndvi.py S2_image.tif output_ndvi.tif 4 8")
