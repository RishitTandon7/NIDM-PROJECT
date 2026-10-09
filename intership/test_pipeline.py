import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from run_comparison import main as run_main

def create_dummy_image(filename, bands=5, width=100, height=100):
    """Create a dummy multi-band GeoTIFF."""
    transform = from_origin(0, 0, 1, 1)
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': bands,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': transform
    }
    
    data = np.random.rand(bands, height, width).astype('float32')
    
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(data)
        
    return filename

def test_pipeline():
    print("Setting up test environment...")
    base_dir = r"f:\NIDM-SRM Drone Project Docs\Data"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    # Create dummy images for L8, S2, LISS
    # L8: 5 bands (Blue, Green, Red, NIR, SWIR)
    l8_path = os.path.join(base_dir, 'L8_image.tif')
    create_dummy_image(l8_path, bands=5)
    
    # S2: 5 bands
    s2_path = os.path.join(base_dir, 'S2_image.tif')
    create_dummy_image(s2_path, bands=5)
    
    # LISS: 3 bands (Green, Red, NIR)
    liss_path = os.path.join(base_dir, 'LISS_image.tif')
    create_dummy_image(liss_path, bands=3)
    
    print("Dummy images created. Running pipeline...")
    
    try:
        run_main()
        print("\nPipeline execution SUCCESSFUL.")
    except Exception as e:
        print(f"\nPipeline execution FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup (optional, maybe keep for user to see?)
        # os.remove(l8_path)
        # os.remove(s2_path)
        # os.remove(liss_path)
        pass

if __name__ == "__main__":
    test_pipeline()





