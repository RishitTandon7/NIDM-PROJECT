import numpy as np
import matplotlib.pyplot as plt
from data_processing import preprocess_dem
from dem_analysis import (calculate_slope, calculate_aspect, calculate_curvature,
                         calculate_plan_curvature, calculate_elevation, calculate_hillshade)
from indices_calculation import (calculate_twi, calculate_spi, calculate_distance_to_glacier,
                                calculate_distance_to_lake, calculate_flow_accumulation,
                                calculate_catchment_area, calculate_compound_topographic_index,
                                calculate_terrain_ruggedness_index, calculate_vector_ruggedness_measure)
from ml_classification import (prepare_features, prepare_labels, train_decision_tree,
                              train_random_forest, train_svm, cross_validate_model, predict_glacier_probability)

def create_synthetic_dem(size=(100, 100)):
    """Create synthetic DEM for testing."""
    x, y = np.meshgrid(np.linspace(0, 10, size[1]), np.linspace(0, 10, size[0]))
    dem = 1000 + 50 * np.sin(x/2) + 30 * np.cos(y/3) + np.random.normal(0, 5, size)
    return dem.astype(np.float32)

def create_synthetic_glacier_mask(dem):
    """Create synthetic glacier mask."""
    # Simple mask: areas with elevation > 1020 are glaciers
    glacier_mask = (dem > 1020).astype(np.uint8)
    return glacier_mask

def create_mock_profile(dem):
    """Create mock profile for synthetic data."""
    return {
        'transform': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),  # Simple 1m resolution
        'crs': 'EPSG:4326',
        'width': dem.shape[1],
        'height': dem.shape[0]
    }

def plot_results(dem, slope, aspect, glacier_mask, curvature=None, predictions=None):
    """Plot analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0,0].imshow(dem, cmap='terrain')
    axes[0,0].set_title('DEM')

    axes[0,1].imshow(slope, cmap='Reds')
    axes[0,1].set_title('Slope (degrees)')

    axes[0,2].imshow(aspect, cmap='hsv')
    axes[0,2].set_title('Aspect (degrees)')

    axes[1,0].imshow(glacier_mask, cmap='Blues')
    axes[1,0].set_title('Glacier Mask')

    if curvature is not None:
        axes[1,1].imshow(curvature, cmap='RdYlBu')
        axes[1,1].set_title('Curvature')
    else:
        axes[1,1].axis('off')

    if predictions is not None:
        axes[1,2].imshow(predictions.reshape(dem.shape), cmap='Greens', alpha=0.7)
        axes[1,2].set_title('Predictions')
    else:
        axes[1,2].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    print("Running GLOF analysis with synthetic data for demonstration...")

    try:
        # Create synthetic data
        print("Creating synthetic DEM...")
        dem = create_synthetic_dem()
        profile = create_mock_profile(dem)

        print("Preprocessing DEM...")
        preprocessed_dem = preprocess_dem(dem)

        # Calculate DEM attributes
        print("Calculating DEM attributes...")
        slope = calculate_slope(preprocessed_dem, profile)
        aspect = calculate_aspect(preprocessed_dem, profile)
        curvature = calculate_curvature(preprocessed_dem, profile)
        plan_curvature = calculate_plan_curvature(preprocessed_dem, profile)
        elevation = calculate_elevation(preprocessed_dem)
        hillshade = calculate_hillshade(preprocessed_dem, profile)

        # Calculate indices
        print("Calculating topographic indices...")
        twi = calculate_twi(slope, profile)
        spi = calculate_spi(slope, profile)
        # For synthetic data, create simple distance arrays
        distance_to_glacier = np.random.rand(*dem.shape) * 100
        distance_to_lake = np.random.rand(*dem.shape) * 200
        flow_acc = calculate_flow_accumulation(preprocessed_dem, profile)
        catchment_area = calculate_catchment_area(preprocessed_dem, profile)
        cti = calculate_compound_topographic_index(slope, aspect)
        tri = calculate_terrain_ruggedness_index(preprocessed_dem, profile)
        vrm = calculate_vector_ruggedness_measure(preprocessed_dem, profile)

        # Create labels
        print("Creating glacier mask...")
        glacier_mask = create_synthetic_glacier_mask(preprocessed_dem)

        # Prepare features and labels
        print("Preparing features and labels...")
        features, valid_mask = prepare_features(slope, aspect, curvature, twi, spi, distance_to_glacier,
                                               elevation, plan_curvature)
        labels = prepare_labels(glacier_mask, valid_mask)

        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Glacier pixels: {np.sum(labels)}")

        # Train models
        print("Training Decision Tree model...")
        dt_model = train_decision_tree(features, labels)

        print("Training Random Forest model...")
        rf_model = train_random_forest(features, labels)

        print("Training SVM model...")
        svm_model, scaler = train_svm(features, labels)

        # Cross-validation
        print("Performing cross-validation...")
        cv_scores = cross_validate_model(features, labels, model='dt')

        # Plot results
        print("Plotting results...")
        plot_results(preprocessed_dem, slope, aspect, glacier_mask)

        print("GLOF analysis with synthetic data completed successfully!")
        print("\nTo run with real data, update the file paths in the main() function.")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()




