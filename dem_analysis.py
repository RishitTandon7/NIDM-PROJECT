import numpy as np
import rasterio
from scipy import ndimage

def calculate_slope(dem, profile):
    """Calculate slope from DEM using numpy gradient."""
    # Get pixel size from profile
    pixel_size = profile['transform'][0]  # Assuming square pixels
    
    # Calculate gradients
    dz_dx = np.gradient(dem, axis=1) / pixel_size
    dz_dy = np.gradient(dem, axis=0) / pixel_size
    
    # Calculate slope in degrees
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)) * (180 / np.pi)
    return slope

def calculate_aspect(dem, profile):
    """Calculate aspect from DEM."""
    pixel_size = profile['transform'][0]
    
    dz_dx = np.gradient(dem, axis=1) / pixel_size
    dz_dy = np.gradient(dem, axis=0) / pixel_size
    
    aspect = np.arctan2(dz_dy, dz_dx) * (180 / np.pi)
    # Convert to 0-360 range
    aspect = (aspect + 360) % 360
    return aspect

def calculate_curvature(dem, profile):
    """Calculate profile curvature from DEM."""
    pixel_size = profile['transform'][0]
    
    # Second derivatives
    d2z_dx2 = np.gradient(np.gradient(dem, axis=1), axis=1) / (pixel_size**2)
    d2z_dy2 = np.gradient(np.gradient(dem, axis=0), axis=0) / (pixel_size**2)
    
    curvature = d2z_dx2 + d2z_dy2
    return curvature

def calculate_plan_curvature(dem, profile):
    """Calculate plan curvature."""
    pixel_size = profile['transform'][0]
    
    dz_dx = np.gradient(dem, axis=1) / pixel_size
    dz_dy = np.gradient(dem, axis=0) / pixel_size
    d2z_dx2 = np.gradient(dz_dx, axis=1) / pixel_size
    d2z_dy2 = np.gradient(dz_dy, axis=0) / pixel_size
    d2z_dxdy = np.gradient(dz_dx, axis=0) / pixel_size
    
    p = dz_dx**2 + dz_dy**2
    plan_curvature = (d2z_dx2 * dz_dy**2 - 2 * d2z_dxdy * dz_dx * dz_dy + d2z_dy2 * dz_dx**2) / (p * np.sqrt(p + 1))
    return plan_curvature

def calculate_elevation(dem):
    """Return elevation (DEM itself)."""
    return dem

def calculate_hillshade(dem, profile, azimuth=315, altitude=45):
    """Calculate hillshade."""
    pixel_size = profile['transform'][0]
    
    dz_dx = np.gradient(dem, axis=1) / pixel_size
    dz_dy = np.gradient(dem, axis=0) / pixel_size
    
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(dz_dy, dz_dx)
    
    hillshade = (np.cos(np.radians(altitude)) * np.cos(slope) + 
                 np.sin(np.radians(altitude)) * np.sin(slope) * np.cos(np.radians(azimuth) - aspect))
    hillshade = np.clip(hillshade, 0, 1) * 255
    return hillshade.astype(np.uint8)

# Add more DEM analysis functions as needed



