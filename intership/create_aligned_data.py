"""
Create synthetic satellite data that matches the shapefile extent
This allows training to work properly while you obtain real satellite imagery
"""
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os


def create_synthetic_satellite_data(shp_bounds, output_path, n_bands=5, resolution=0.0003):
    """
    Create synthetic satellite-like data covering the shapefile extent.
    
    Args:
        shp_bounds: (left, bottom, right, top) bounds from shapefile
        output_path: Where to save the synthetic TIFF
        n_bands: Number of spectral bands
        resolution: Spatial resolution in degrees (~30m at equator)
    """
    
    left, bottom, right, top = shp_bounds
    
    # Calculate dimensions
    width = int((right - left) / resolution)
    height = int((top - bottom) / resolution)
    
    print(f"Creating synthetic data:")
    print(f"  Extent: {shp_bounds}")
    print(f"  Resolution: {resolution}° (~{resolution*111000:.0f}m)")
    print(f"  Dimensions: {width} x {height} pixels")
    print(f"  Bands: {n_bands}")
    
    # Create transform
    transform = from_bounds(left, bottom, right, top, width, height)
    
    # Generate synthetic data with some spatial structure
    np.random.seed(42)
    
    data = []
    for band_idx in range(n_bands):
        # Create base pattern
        base = np.random.rand(height, width) * 0.5 + 0.25
        
        # Add some spatial correlation (smooth patterns)
        from scipy.ndimage import gaussian_filter
        band_data = gaussian_filter(base, sigma=5)
        
        # Normalize to 0-1 range
        band_data = (band_data - band_data.min()) / (band_data.max() - band_data.min())
        
        data.append(band_data.astype(np.float32))
    
    data = np.array(data)
    
    # Save as GeoTIFF
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': n_bands,
        'height': height,
        'width': width,
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'lzw'
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)
    
    print(f"[OK] Saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    return data, profile


def main():
    import fiona
    
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    shp_path = os.path.join(base_dir, 'GLOF project', 'image analysis', 'classified_glacier_dt.shp')
    
    # Get shapefile bounds
    with fiona.open(shp_path) as shp:
        bounds = shp.bounds
        print(f"\nShapefile bounds: {bounds}")
    
    # Create synthetic data for each satellite type
    data_dir = os.path.join(base_dir, 'Data_Aligned')
    os.makedirs(data_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("CREATING ALIGNED SYNTHETIC SATELLITE DATA")
    print("="*70 + "\n")
    
    # Landsat 8 (7 bands but we'll use 5)
    print("\n### Creating Landsat 8 synthetic data ###")
    create_synthetic_satellite_data(
        shp_bounds=bounds,
        output_path=os.path.join(data_dir, 'L8_image_aligned.tif'),
        n_bands=5,
        resolution=0.0003  # ~30m
    )
    
    # Sentinel-2 (fewer bands)
    print("\n### Creating Sentinel-2 synthetic data ###")
    create_synthetic_satellite_data(
        shp_bounds=bounds,
        output_path=os.path.join(data_dir, 'S2_image_aligned.tif'),
        n_bands=5,
        resolution=0.0001  # ~10m
    )
    
    # LISS-IV
    print("\n### Creating LISS-IV synthetic data ###")
    create_synthetic_satellite_data(
        shp_bounds=bounds,
        output_path=os.path.join(data_dir, 'LISS_image_aligned.tif'),
        n_bands=3,
        resolution=0.00006  # ~5.8m
    )
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nSynthetic data saved to: {data_dir}")
    print("\nNOTE: This is SYNTHETIC data for testing the pipeline.")
    print("For real results, you need actual satellite imagery covering:")
    print(f"  Longitude: {bounds[0]:.4f}° to {bounds[2]:.4f}°E")
    print(f"  Latitude: {bounds[1]:.4f}° to {bounds[3]:.4f}°N")
    print("  (Sikkim Himalayas region)")


if __name__ == "__main__":
    main()





