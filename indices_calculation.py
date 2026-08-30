import numpy as np
from scipy import ndimage
from skimage import morphology
import geopandas as gpd
from rasterio import features
from shapely.geometry import Point

def calculate_twi(slope, profile):
    """Calculate Topographic Wetness Index (TWI)."""
    # Simplified TWI calculation
    # In practice, this requires flow accumulation (catchment area)
    # For now, using a simplified version
    twi = np.log(1 / np.tan(np.radians(slope + 0.1)))  # Add small value to avoid division by zero
    return twi

def calculate_spi(slope, profile):
    """Calculate Stream Power Index (SPI)."""
    # SPI = A * tan(beta) where A is specific catchment area, beta is slope
    # Simplified: assuming unit catchment area
    spi = np.tan(np.radians(slope))
    return spi

def calculate_distance_to_glacier(dem, glacier_shp_path, profile):
    """Calculate distance to glacier."""
    # Load glacier shapefile
    glacier_gdf = gpd.read_file(glacier_shp_path)
    
    # Create glacier mask
    glacier_mask = features.rasterize(
        [(geom, 1) for geom in glacier_gdf.geometry],
        out_shape=dem.shape,
        transform=profile['transform'],
        fill=0,
        dtype=np.uint8
    )
    
    # Calculate distance transform
    distance = ndimage.distance_transform_edt(1 - glacier_mask)
    return distance

def calculate_distance_to_lake(dem, lake_shp_path, profile):
    """Calculate distance to lake."""
    lake_gdf = gpd.read_file(lake_shp_path)
    
    lake_mask = features.rasterize(
        [(geom, 1) for geom in lake_gdf.geometry],
        out_shape=dem.shape,
        transform=profile['transform'],
        fill=0,
        dtype=np.uint8
    )
    
    distance = ndimage.distance_transform_edt(1 - lake_mask)
    return distance

def calculate_flow_accumulation(dem, profile):
    """Calculate flow accumulation (simplified)."""
    # This is a very simplified version. In practice, use proper flow routing algorithms
    flow_dir = ndimage.generic_filter(dem, lambda x: np.argmin(x), size=3, mode='constant', cval=np.inf)
    flow_acc = np.ones_like(dem)
    # Placeholder for actual flow accumulation
    return flow_acc

def calculate_catchment_area(dem, profile):
    """Calculate specific catchment area."""
    # Simplified
    return calculate_flow_accumulation(dem, profile)

def calculate_compound_topographic_index(slope, aspect):
    """Calculate Compound Topographic Index (CTI)."""
    cti = np.log(1 / np.tan(np.radians(slope + 0.1))) * np.cos(np.radians(aspect))
    return cti

def calculate_terrain_ruggedness_index(dem, profile):
    """Calculate Terrain Ruggedness Index (TRI)."""
    # TRI is the mean difference between a central pixel and its surrounding cells
    kernel = np.ones((3,3)) / 8
    mean_elev = ndimage.convolve(dem, kernel, mode='nearest')
    tri = np.sqrt(np.sum((dem - mean_elev)**2, axis=(0,1) if dem.ndim == 3 else None))
    return tri

def calculate_vector_ruggedness_measure(dem, profile):
    """Calculate Vector Ruggedness Measure (VRM)."""
    # VRM measures terrain ruggedness in 3D
    dz_dx = np.gradient(dem, axis=1)
    dz_dy = np.gradient(dem, axis=0)
    
    vrm = np.sqrt(dz_dx**2 + dz_dy**2 + 1)
    return vrm

# Add more index calculations as needed from the notebook



