import numpy as np
from data_processing import load_dem, load_shapefile, clip_dem_to_shapefile
from dem_analysis import calculate_slope, calculate_aspect, calculate_curvature
from indices_calculation import calculate_twi, calculate_spi, calculate_distance_to_glacier
from ml_classification import prepare_features, prepare_labels, train_decision_tree

def main():
    # Define paths (placeholders, adjust as needed)
    dem_path = 'path/to/dem.tif'
    shp_path = 'path/to/shapefile.shp'
    glacier_shp_path = 'path/to/glacier.shp'
    
    # Load data
    dem, profile = load_dem(dem_path)
    gdf = load_shapefile(shp_path)
    glacier_gdf = load_shapefile(glacier_shp_path)
    
    # Clip DEM
    clipped_dem, clipped_profile = clip_dem_to_shapefile(dem, profile, gdf)
    
    # Calculate DEM attributes
    slope = calculate_slope(clipped_dem, clipped_profile)
    aspect = calculate_aspect(clipped_dem, clipped_profile)
    curvature = calculate_curvature(clipped_dem, clipped_profile)
    
    # Calculate indices
    twi = calculate_twi(slope, clipped_profile)
    spi = calculate_spi(clipped_dem, clipped_profile)
    distance = calculate_distance_to_glacier(clipped_dem, glacier_gdf)
    
    # Prepare features and labels
    features = prepare_features(slope, aspect, curvature, twi, spi, distance)
    # Assuming glacier_mask is created from glacier_gdf
    glacier_mask = np.zeros_like(clipped_dem)  # Placeholder, need to create actual mask
    labels = prepare_labels(glacier_mask)
    
    # Train model
    clf = train_decision_tree(features, labels)
    
    print("GLOF analysis completed.")

if __name__ == "__main__":
    main()




