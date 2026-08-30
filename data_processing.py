import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import mapping

def load_dem(dem_path):
    """Load DEM data from file."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        profile = src.profile
    return dem, profile

def load_shapefile(shp_path):
    """Load shapefile data."""
    gdf = gpd.read_file(shp_path)
    return gdf

def clip_dem_to_shapefile(dem_path, shp_path):
    """Clip DEM to shapefile boundaries."""
    with rasterio.open(dem_path) as src:
        gdf = gpd.read_file(shp_path)
        geom = [mapping(gdf.geometry.iloc[0])]  # Assuming single geometry
        clipped_dem, clipped_transform = mask(src, geom, crop=True)
        clipped_profile = src.profile.copy()
        clipped_profile.update({
            "height": clipped_dem.shape[1],
            "width": clipped_dem.shape[2],
            "transform": clipped_transform
        })
    return clipped_dem[0], clipped_profile  # Return the array and profile

def reproject_dem(dem, src_profile, dst_crs):
    """Reproject DEM to a different CRS."""
    transform, width, height = calculate_default_transform(
        src_profile['crs'], dst_crs, src_profile['width'], src_profile['height'], *src_profile['transform'].bounds
    )
    dst_profile = src_profile.copy()
    dst_profile.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    dst_dem = np.zeros((height, width), dtype=dem.dtype)
    reproject(
        source=dem,
        destination=dst_dem,
        src_transform=src_profile['transform'],
        src_crs=src_profile['crs'],
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear
    )
    return dst_dem, dst_profile

def load_glacier_shapefile(glacier_shp_path):
    """Load glacier shapefile."""
    return load_shapefile(glacier_shp_path)

def load_lake_shapefile(lake_shp_path):
    """Load lake shapefile."""
    return load_shapefile(lake_shp_path)

def preprocess_dem(dem):
    """Preprocess DEM: fill sinks, etc. (placeholder)."""
    # Placeholder for sink filling
    return dem

# Add more data processing functions as needed from the notebook



