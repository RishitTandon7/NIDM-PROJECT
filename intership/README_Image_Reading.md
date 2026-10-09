# Reading Satellite Images with Rasterio/GDAL - Summary

Successfully demonstrated reading satellite imagery using **rasterio** library!

## What Was Done

1. **Created comprehensive image reading script**: [read_satellite_images.py](file:///f:/NIDM-SRM%20Drone%20Project%20Docs/intership/read_satellite_images.py)
2. **Read satellite images** from the Data folder using rasterio
3. **Generated visualizations** showing individual bands and RGB composites

## Key Capabilities Demonstrated

### 1. Reading Image Data with Rasterio

```python
import rasterio

# Open and read satellite image
with rasterio.open('satellite_image.tif') as src:
    # Read metadata
    print(f"Dimensions: {src.width} x {src.height}")
    print(f"Number of bands: {src.count}")
    print(f"CRS: {src.crs}")
    print(f"Transform: {src.transform}")
    
    # Read all bands: shape = (bands, height, width)
    data = src.read()
```

### 2. Accessing Metadata

The script shows how to access important metadata:
- **Driver**: Image format (GTiff, VRT, etc.)
- **Dimensions**: Width x Height in pixels
- **Bands**: Number of spectral bands
- **CRS**: Coordinate Reference System
- **Transform**: Geotransform matrix
- **Bounds**: Geographic bounding box
- **Data type**: numpy dtype of pixel values

### 3. Band Visualization

Generated individual band visualizations for:
- [L8_bands_visualization.png](file:///f:/NIDM-SRM%20Drone%20Project%20Docs/intership/L8_bands_visualization.png) - Landsat 8 bands
- [S2_bands_visualization.png](file:///f:/NIDM-SRM%20Drone%20Project%20Docs/intership/S2_bands_visualization.png) - Sentinel-2 bands

### 4. RGB Composites

Created false-color composites for visualization:
- [L8_rgb_composite.png](file:///f:/NIDM-SRM%20Drone%20Project%20Docs/intership/L8_rgb_composite.png) - Landsat 8 RGB
- [S2_rgb_composite.png](file:///f:/NIDM-SRM%20Drone%20Project%20Docs/intership/S2_rgb_composite.png) - Sentinel-2 RGB

## VRT File Handling

**Important Note**: VRT (Virtual Raster) files are XML files that reference other raster files. They don't contain actual image data.

Your VRT files reference source TIFF files at:
```
F:\CTECH\Research\proposal\NIDM\GLOF data\Landsat 5\
```

These files don't exist on your current system, which is why the VRT reading fails. VRT files are useful for:
- Mosaicking multiple tiles
- Creating virtual band stacks
- Referencing subsets of large files
- Warping without creating new files

## Available Satellite Images

The script found these images in your Data folder:
- `L8_image.tif` (0.19 MB) - Landsat 8
- `LISS_image.tif` (0.11 MB) - LISS-IV
- `S2_image.tif` (0.19 MB) - Sentinel-2

All images are 100x100 pixels with 5 bands in float32 format.

## How to Use

Run the demonstration script:
```powershell
cd "f:\NIDM-SRM Drone Project Docs\intership"
python read_satellite_images.py
```

Or use the functions in your own code:
```python
from read_satellite_images import read_with_rasterio, visualize_bands, create_rgb_composite

# Read image
data, metadata = read_with_rasterio('path/to/image.tif')

# Visualize bands
visualize_bands(data, metadata, output_path='bands.png')

# Create RGB composite
create_rgb_composite(data, rgb_bands=(3, 2, 1), output_path='rgb.png')
```

## Integration with Training Pipeline

The existing training code already uses rasterio through [satellite_processing.py](file:///f:/NIDM-SRM%20Drone%20Project%20Docs/intership/satellite_processing.py):

```python
def load_satellite_image(filepath):
    with rasterio.open(filepath) as src:
        data = src.read()
        profile = src.profile
    return data, profile
```

This is the same approach demonstrated in the new script!
