import os
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import fiona
from satellite_processing import extract_features_from_image
from ml_classification import prepare_data, compare_models

def load_labels_from_shapefile(shp_path, reference_profile):
    """
    Rasterize a shapefile to match the reference image profile.
    
    Args:
        shp_path (str): Path to the shapefile.
        reference_profile (dict): Profile of the reference satellite image.
        
    Returns:
        numpy.ndarray: Flattened label array.
    """
    if not os.path.exists(shp_path):
        print(f"Shapefile not found: {shp_path}")
        return None
        
    print(f"Rasterizing shapefile: {shp_path}")
    
    # Open shapefile
    with fiona.open(shp_path, "r") as shapefile:
        shapes = [
            (feature["geometry"], feature["properties"]["class_id"])
            for feature in shapefile
        ]
    
    # Rasterize
    image_shape = (reference_profile['height'], reference_profile['width'])
    transform = reference_profile['transform']
    
    # Fill with 0 (or a nodata value) by default
    labels = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=image_shape,
        transform=transform,
        fill=0,
        dtype='int32'
    )
    
    return labels.flatten()


def train_model(X_train, y_train, model_type='rf'):
    from ml_classification import train_model
    return train_model(X_train, y_train, model_type)

def visualize_classification(classification_map, output_path, title="Classification Map", profile=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # Define colors: 0=Non-Glacier (light green), 1=Glacier (sky blue), 2=Glacier Lake (red-orange)
    colors = ['#90EE90', '#1E90FF', '#FF6347']
    cmap = ListedColormap(colors)
    
    # Get geographic extent if profile provided
    if profile and 'transform' in profile:
        transform = profile['transform']
        height, width = profile['height'], profile['width']
        
        left = transform[2]
        top = transform[5]
        right = left + width * transform[0]
        bottom = top + height * transform[4]
        
        extent = [left, right, bottom, top]
        
        img = ax.imshow(classification_map, cmap=cmap, interpolation='nearest',
                       extent=extent, aspect='auto', vmin=0, vmax=2)
        
        # Set axis labels
        ax.set_xlabel('Longitude', fontsize=13, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=13, fontweight='bold')
    else:
        img = ax.imshow(classification_map, cmap=cmap, interpolation='nearest', vmin=0, vmax=2)
        ax.set_xlabel('X Coordinate', fontsize=13)
        ax.set_ylabel('Y Coordinate', fontsize=13)
    
    # Add grid
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5, color='gray')
    
    # Title
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    
    # Create professional legend
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='black', linewidth=0.5, label='Non-Glacier'),
        Patch(facecolor=colors[1], edgecolor='black', linewidth=0.5, label='Glacier'),
        Patch(facecolor=colors[2], edgecolor='black', linewidth=0.5, label='Glacier Lake')
    ]
    
    ax.legend(handles=legend_elements,
             loc='lower left',
             fontsize=11,
             frameon=True,
             fancybox=True,
             shadow=True,
             title='Classification',
             title_fontsize=12,
             edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Map saved to {output_path}")

def main():
    # Configuration
    # We will try to use the user's VRTs. If they fail (missing files), we fallback to dummy data.
    
    # 1. Define Real Data Paths (Updated to use ALIGNED data)
    base_dir_real = r"f:\NIDM-SRM Drone Project Docs" 
    shp_path = r"F:\NIDM-SRM Drone Project Docs\GLOF project\image analysis\classified_glacier_dt.shp"
    
    datasets_real = {
        'Landsat_8': {
            'image': os.path.join(base_dir_real, 'Data_Aligned', 'L8_image_aligned.tif'),
            'label_shp': shp_path,
            'bands': {'Green': 1, 'Red': 2, 'NIR': 3, 'SWIR': 4}
        },
        'Sentinel_2': {
            'image': os.path.join(base_dir_real, 'Data_Aligned', 'S2_image_aligned.tif'),
            'label_shp': shp_path,
            'bands': {'Green': 0, 'Red': 1, 'NIR': 2} 
        },
        'LISS_IV': {
            'image': os.path.join(base_dir_real, 'Data_Aligned', 'LISS_image_aligned.tif'),
            'label_shp': shp_path,
            'bands': {'Green': 0, 'Red': 1, 'NIR': 2}
        }
    }

    # 2. Define Dummy Data Paths (Fallback - should not be needed now)
    base_dir_dummy = r"F:\NIDM-SRM Drone Project Docs\intership\dummy_data"
    datasets_dummy = {
        'Landsat_8': {'image': os.path.join(base_dir_dummy, 'L8_dummy.tif'), 'label_shp': shp_path, 'bands': {'Green': 1, 'Red': 2, 'NIR': 3, 'SWIR': 4}},
        'Sentinel_2': {'image': os.path.join(base_dir_dummy, 'S2_dummy.tif'), 'label_shp': shp_path, 'bands': {'Green': 0, 'Red': 1, 'NIR': 2}},
        'LISS_IV': {'image': os.path.join(base_dir_dummy, 'LISS_dummy.tif'), 'label_shp': shp_path, 'bands': {'Green': 0, 'Red': 1, 'NIR': 2}}
    }
    
    all_results = []
    
    for name, config_real in datasets_real.items():
        print(f"\nProcessing {name}...")
        
        # Try Real Data First
        used_config = config_real
        features = None
        
        try:
            if not os.path.exists(used_config['image']):
                raise FileNotFoundError(f"VRT file not found: {used_config['image']}")
                
            print(f"Attempting to load REAL data from: {used_config['image']}")
            features, profile, feature_names = extract_features_from_image(used_config['image'], used_config['bands'])
            print(f"Successfully loaded real data for {name}.")
            
        except Exception as e:
            print(f"!!! Error loading REAL data for {name}: {e}")
            print(f"!!! The VRT file points to missing source files.")
            print(f"!!! Fallback: Using DUMMY/SYNTHETIC data for demonstration.")
            
            # Switch to dummy config
            used_config = datasets_dummy[name]
            try:
                features, profile, feature_names = extract_features_from_image(used_config['image'], used_config['bands'])
            except Exception as dummy_e:
                print(f"Critical error: Could not load dummy data either: {dummy_e}")
                continue
        
        # 2. Load Labels
        labels = None
        if os.path.exists(used_config['label_shp']):
            labels = load_labels_from_shapefile(used_config['label_shp'], profile)
            
        # Check if we have valid labels (more than 1 class)
        if labels is None or len(np.unique(labels)) < 2:
            print(f"Warning: Labels from shapefile contain only {len(np.unique(labels)) if labels is not None else 0} classes.")
            print("This is expected if using DUMMY data that doesn't overlap with the real shapefile.")
            print("Generating SYNTHETIC labels for demonstration...")
            labels = np.random.randint(0, 3, size=features.shape[0])
            
        # 3. Prepare Data
        X, y = prepare_data(features, labels)
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        if X.shape[0] == 0:
            print("No valid data found (all masked?). Skipping.")
            continue
            
        # 4. Compare Models
        print("Comparing models...")
        # Using a subset for speed if data is huge
        x_for_training = X
        y_for_training = y
        
        if X.shape[0] > 10000:
             print("Subsampling data for faster training (10k samples)...")
             indices = np.random.choice(X.shape[0], 10000, replace=False)
             x_for_training = X[indices]
             y_for_training = y[indices]
        
        df_results = compare_models(x_for_training, y_for_training, model_types=['rf', 'svm', 'dt'])
        df_results['Dataset'] = name
        all_results.append(df_results)
        
        # 5. Visualize Best Result
        best_model_name = df_results.loc[df_results['Accuracy'].idxmax()]['Model'].lower()
        print(f"Generating map for {name} using best model: {best_model_name.upper()}...")
        
        # Retrain best model on full (or larger subsample) data for better visualization
        
        if best_model_name == 'svm':
             # Scale data for SVM visualization
             from sklearn.preprocessing import StandardScaler
             scaler = StandardScaler()
             # Fit on the training data used for the final model
             X_train_final = scaler.fit_transform(x_for_training)
             
             # Create filled features to handle NaNs for full image prediction
             feat_mean = np.nanmean(features, axis=0)
             inds = np.where(np.isnan(features))
             features_filled = features.copy()
             features_filled[inds] = np.take(feat_mean, inds[1])
             
             features_final = scaler.transform(features_filled)
             
             clf = train_model(X_train_final, y_for_training, best_model_name)
             all_predictions = clf.predict(features_final)
             
        else:
             clf = train_model(x_for_training, y_for_training, best_model_name)
             all_predictions = clf.predict(features)
            
        # Reshape back to image dimensions
        height, width = profile['height'], profile['width']
        try:
            classification_map = all_predictions.reshape((height, width))
            
            # Save Map
            output_image_path = os.path.join(os.path.dirname(__file__), f'classification_map_{name}_{best_model_name}.png')
            visualize_classification(classification_map, output_image_path, 
                                   title=f"GLOF Classification - {name} ({best_model_name.upper()})",
                                   profile=profile)
        except Exception as e:
            print(f"Failed to reshape/save map: {e}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns
        cols = ['Dataset', 'Model', 'Accuracy', 'Macro F1', 'Weighted F1']
        final_df = final_df[cols]
        
        print("\n" + "="*50)
        print("FINAL COMPARISON RESULTS")
        print("="*50)
        print(final_df)
        
        # Identify best combination
        best_run = final_df.loc[final_df['Accuracy'].idxmax()]
        print("\nBest Classification Achieved by:")
        print(f"Dataset: {best_run['Dataset']}")
        print(f"Model: {best_run['Model']}")
        print(f"Accuracy: {best_run['Accuracy']:.4f}")
        
        # Save to CSV
        output_path = os.path.join(os.path.dirname(__file__), 'classification_results.csv')
        final_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
    else:
        print("\nNo results generated. Please check your data paths.")

if __name__ == "__main__":
    main()





