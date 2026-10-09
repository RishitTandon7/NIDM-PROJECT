import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

def create_dummy_image(filename, bands, width=500, height=500):
    """Create a dummy multi-band GeoTIFF with random data."""
    print(f"Creating dummy image: {filename}")
    transform = from_origin(87.0, 27.0, 0.0003, 0.0003) # Approx coords
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': bands,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': transform
    }
    
    # Generate random data (0-1 range for reflectance)
    data = np.random.rand(bands, height, width).astype('float32')
    
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(data)

def main():
    output_dir = r"F:\NIDM-SRM Drone Project Docs\intership\dummy_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Landsat 5 (7 bands)
    create_dummy_image(os.path.join(output_dir, "L5_dummy.tif"), bands=7)
    
    # Landsat 8 (7 bands for our subset)
    create_dummy_image(os.path.join(output_dir, "L8_dummy.tif"), bands=7)
    
    # Sentinel 2 (4 bands: B, G, R, NIR)
    create_dummy_image(os.path.join(output_dir, "S2_dummy.tif"), bands=4)
    
    # LISS IV (3 bands: G, R, NIR)
    create_dummy_image(os.path.join(output_dir, "LISS_dummy.tif"), bands=3)
    
    print("\nDummy data generation complete.")

if __name__ == "__main__":
    main()





