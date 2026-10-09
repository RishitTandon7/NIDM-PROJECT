import rasterio
import numpy as np
from rasterio.enums import Resampling

def load_satellite_image(filepath):
    """
    Load a satellite image from a GeoTIFF file.
    
    Args:
        filepath (str): Path to the GeoTIFF file.
        
    Returns:
        numpy.ndarray: Image data (bands, height, width).
        dict: Profile metadata of the image.
    """
    with rasterio.open(filepath) as src:
        data = src.read()
        profile = src.profile
    return data, profile

def calculate_ndvi(red_band, nir_band):
    """
    Calculate Normalized Difference Vegetation Index (NDVI).
    NDVI = (NIR - Red) / (NIR + Red)
    """
    numerator = nir_band - red_band
    denominator = nir_band + red_band
    # Avoid division by zero
    ndvi = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator!=0)
    return ndvi

def calculate_ndwi(green_band, nir_band):
    """
    Calculate Normalized Difference Water Index (NDWI).
    NDWI = (Green - NIR) / (Green + NIR)
    """
    numerator = green_band - nir_band
    denominator = green_band + nir_band
    ndwi = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator!=0)
    return ndwi

def calculate_mndwi(green_band, swir_band):
    """
    Calculate Modified Normalized Difference Water Index (MNDWI).
    MNDWI = (Green - SWIR) / (Green + SWIR)
    """
    numerator = green_band - swir_band
    denominator = green_band + swir_band
    mndwi = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator!=0)
    return mndwi

def stack_bands(bands_dict):
    """
    Stack individual bands into a single array.
    
    Args:
        bands_dict (dict): Dictionary where keys are band names and values are 2D numpy arrays.
        
    Returns:
        numpy.ndarray: Stacked array (n_bands, height, width).
    """
    return np.stack(list(bands_dict.values()))

def extract_features_from_image(image_path, band_indices):
    """
    Load image and extract specified bands and calculate indices.
    
    Args:
        image_path (str): Path to the multi-band image.
        band_indices (dict): Dictionary mapping band names to their index (0-based) in the image file.
                             e.g., {'Green': 1, 'Red': 2, 'NIR': 3, 'SWIR': 4}
                             
    Returns:
        numpy.ndarray: Feature matrix (n_pixels, n_features).
    """
    data, profile = load_satellite_image(image_path)
    
    features = {}
    
    # Extract raw bands
    for band_name, idx in band_indices.items():
        if idx < data.shape[0]:
            features[band_name] = data[idx]
        else:
            print(f"Warning: Band index {idx} for {band_name} is out of bounds.")
            
    # Calculate indices if possible
    if 'Green' in features and 'NIR' in features:
        features['NDWI'] = calculate_ndwi(features['Green'], features['NIR'])
        
    if 'Red' in features and 'NIR' in features:
        features['NDVI'] = calculate_ndvi(features['Red'], features['NIR'])
        
    if 'Green' in features and 'SWIR' in features:
        features['MNDWI'] = calculate_mndwi(features['Green'], features['SWIR'])
        
    # Stack features
    feature_stack = np.stack(list(features.values()), axis=-1) # (height, width, n_features)
    
    # Flatten for ML
    n_pixels = feature_stack.shape[0] * feature_stack.shape[1]
    n_features = feature_stack.shape[2]
    flat_features = feature_stack.reshape(n_pixels, n_features)
    
    return flat_features, profile, list(features.keys())





