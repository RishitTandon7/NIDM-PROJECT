"""
Comprehensive Satellite Image Reading Script
Demonstrates reading satellite images using rasterio
Handles VRT files, regular TIFFs, and various satellite formats
"""
import os
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt


def read_with_rasterio(image_path, verbose=True):
    """
    Read satellite image using rasterio.
    
    Args:
        image_path (str): Path to the image file (TIFF, VRT, etc.)
        verbose (bool): Print detailed information
        
    Returns:
        tuple: (data array, metadata dict)
    """
    print(f"\n{'='*70}")
    print(f"Reading with Rasterio: {os.path.basename(image_path)}")
    print(f"{'='*70}")
    
    try:
        with rasterio.open(image_path) as src:
            # Read metadata
            if verbose:
                print(f"\nMetadata:")
                print(f"  Driver: {src.driver}")
                print(f"  Dimensions: {src.width} x {src.height}")
                print(f"  Number of bands: {src.count}")
                print(f"  Data type: {src.dtypes[0]}")
                print(f"  CRS: {src.crs}")
                print(f"  Bounds: {src.bounds}")
                print(f"  Transform: {src.transform}")
                print(f"  No data value: {src.nodata}")
                
                # Band information
                print(f"\n  Band Information:")
                for i in range(1, src.count + 1):
                    print(f"    Band {i}: {src.dtypes[i-1]}")
            
            # Read all bands
            data = src.read()  # Shape: (bands, height, width)
            
            # Get metadata
            metadata = {
                'driver': src.driver,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': src.dtypes[0],
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,
                'nodata': src.nodata,
                'profile': src.profile
            }
            
            if verbose:
                print(f"\nData Array:")
                print(f"  Shape: {data.shape}")
                print(f"  Min value: {np.nanmin(data):.2f}")
                print(f"  Max value: {np.nanmax(data):.2f}")
                print(f"  Mean value: {np.nanmean(data):.2f}")
            
            return data, metadata
            
    except Exception as e:
        print(f"[ERROR] Failed to read with rasterio: {e}")
        return None, None



def visualize_bands(data, metadata, output_path=None, max_bands=4):
    """
    Visualize satellite image bands.
    
    Args:
        data (numpy.ndarray): Image data array (bands, height, width)
        metadata (dict): Image metadata
        output_path (str): Optional path to save visualization
        max_bands (int): Maximum number of bands to visualize
    """
    num_bands = min(data.shape[0], max_bands)
    
    fig, axes = plt.subplots(1, num_bands, figsize=(4*num_bands, 4))
    
    if num_bands == 1:
        axes = [axes]
    
    for i in range(num_bands):
        band_data = data[i]
        
        # Normalize for visualization
        vmin, vmax = np.percentile(band_data[~np.isnan(band_data)], [2, 98])
        
        im = axes[i].imshow(band_data, cmap='gray', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'Band {i+1}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[OK] Visualization saved to: {os.path.basename(output_path)}")
    else:
        plt.show()
    
    plt.close()


def create_rgb_composite(data, rgb_bands=(2, 1, 0), output_path=None):
    """
    Create RGB composite from satellite image.
    
    Args:
        data (numpy.ndarray): Image data array (bands, height, width)
        rgb_bands (tuple): Indices of bands to use for R, G, B
        output_path (str): Optional path to save composite
    """
    try:
        r_band = data[rgb_bands[0]]
        g_band = data[rgb_bands[1]]
        b_band = data[rgb_bands[2]]
        
        # Normalize each band to 0-1
        def normalize(band):
            valid = band[~np.isnan(band)]
            vmin, vmax = np.percentile(valid, [2, 98])
            normalized = np.clip((band - vmin) / (vmax - vmin), 0, 1)
            return normalized
        
        r_norm = normalize(r_band)
        g_norm = normalize(g_band)
        b_norm = normalize(b_band)
        
        # Stack into RGB
        rgb = np.dstack((r_norm, g_norm, b_norm))
        
        # Visualize
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb)
        plt.title(f'RGB Composite (Bands: R={rgb_bands[0]+1}, G={rgb_bands[1]+1}, B={rgb_bands[2]+1})')
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[OK] RGB composite saved to: {os.path.basename(output_path)}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Failed to create RGB composite: {e}")


def main():
    """Demonstration of reading satellite images."""
    
    print("\n" + "="*70)
    print(" SATELLITE IMAGE READING DEMONSTRATION (Using Rasterio)")
    print("="*70)
    
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    output_dir = os.path.join(base_dir, 'intership')
    
    # Example 1: Read TIFF file
    print("\n\n### EXAMPLE 1: Reading TIFF file ###")
    tiff_path = os.path.join(base_dir, 'Data', 'L8_image.tif')
    
    if os.path.exists(tiff_path):
        # Read with rasterio
        data_rio, meta_rio = read_with_rasterio(tiff_path, verbose=True)
        
        # Visualize bands
        if data_rio is not None:
            visualize_bands(data_rio, meta_rio, 
                          output_path=os.path.join(output_dir, 'L8_bands_visualization.png'))
            
            # Create RGB composite (for Landsat 8: R=NIR, G=Red, B=Green)
            if data_rio.shape[0] >= 4:
                create_rgb_composite(data_rio, rgb_bands=(3, 2, 1),
                                   output_path=os.path.join(output_dir, 'L8_rgb_composite.png'))
    else:
        print(f"[WARNING] File not found: {tiff_path}")
    
    
    # Example 2: Read Sentinel-2 TIFF
    print("\n\n### EXAMPLE 2: Reading Sentinel-2 TIFF ###")
    s2_path = os.path.join(base_dir, 'Data', 'S2_image.tif')
    
    if os.path.exists(s2_path):
        data_s2, meta_s2 = read_with_rasterio(s2_path, verbose=True)
        
        if data_s2 is not None:
            visualize_bands(data_s2, meta_s2,
                          output_path=os.path.join(output_dir, 'S2_bands_visualization.png'))
            
            # Create RGB composite (for Sentinel-2: R=Red, G=Green, B=Blue)
            if data_s2.shape[0] >= 3:
                create_rgb_composite(data_s2, rgb_bands=(1, 0, 2),
                                   output_path=os.path.join(output_dir, 'S2_rgb_composite.png'))
    else:
        print(f"[WARNING] File not found: {s2_path}")
    
    
    # Example 3: Try reading VRT file
    print("\n\n### EXAMPLE 3: Reading VRT file ###")
    vrt_path = os.path.join(base_dir, 'L5_VRT', 'Landsat5_2009-03-14.vrt')
    
    if os.path.exists(vrt_path):
        print(f"\n[INFO] VRT file exists: {vrt_path}")
        print("[INFO] Attempting to read VRT (may fail if source files are missing)...")
        
        # Try reading VRT
        data_vrt, meta_vrt = read_with_rasterio(vrt_path, verbose=False)
        
        if data_vrt is None:
            print("\n[NOTE] VRT reading failed because source TIFF files are missing.")
            print("VRT files are 'virtual' datasets that reference other files.")
            print("The source files should be located at the paths specified in the VRT XML.")
        else:
            print("[OK] VRT file read successfully!")
            visualize_bands(data_vrt, meta_vrt,
                          output_path=os.path.join(output_dir, 'L5_VRT_bands_visualization.png'))
    else:
        print(f"[WARNING] VRT file not found: {vrt_path}")
    
    
    # Example 4: List all available satellite images
    print("\n\n### EXAMPLE 4: Available Satellite Images ###")
    data_dir = os.path.join(base_dir, 'Data')
    
    if os.path.isdir(data_dir):
        print(f"\nAvailable images in Data folder:")
        for file in os.listdir(data_dir):
            if file.endswith(('.tif', '.tiff')):
                filepath = os.path.join(data_dir, file)
                file_size = os.path.getsize(filepath) / (1024*1024)  # MB
                print(f"  - {file} ({file_size:.2f} MB)")
    
    print("\n" + "="*70)
    print(" READING DEMONSTRATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()





