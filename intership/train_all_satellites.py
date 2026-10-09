import os
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import fiona
from satellite_processing import extract_features_from_image
from ml_classification import prepare_data, train_model, evaluate_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pickle

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
    
    # Fill with -1 (nodata) by default
    labels = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=image_shape,
        transform=transform,
        fill=-1,
        dtype='int32'
    )
    
    return labels.flatten()


def visualize_classification(classification_map, output_path, title="Classification Map"):
    """
    Create and save a classification map visualization.
    
    Args:
        classification_map (numpy.ndarray): 2D array of classified pixels.
        output_path (str): Path to save the visualization.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(12, 10))
    
    # Define colors: 0=Non-Glacier(Green), 1=Glacier(Blue), 2=Glacier Lake(Red)
    cmap = ListedColormap(['lightgreen', 'deepskyblue', 'red'])
    
    img = plt.imshow(classification_map, cmap=cmap, interpolation='nearest')
    plt.colorbar(img, ticks=[0, 1, 2], label='Class')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Map saved to {output_path}")


def train_on_satellite_data(name, image_path, shp_path, band_indices, model_types=['rf', 'svm', 'dt']):
    """
    Train and evaluate models on a specific satellite dataset.
    
    Args:
        name (str): Dataset name (e.g., 'Landsat_5').
        image_path (str): Path to the satellite image VRT/TIFF.
        shp_path (str): Path to the training labels shapefile.
        band_indices (dict): Mapping of band names to indices.
        model_types (list): List of model types to train.
        
    Returns:
        dict: Results including trained models, metrics, and predictions.
    """
    print(f"\n{'='*70}")
    print(f"Processing {name}")
    print(f"{'='*70}")
    
    # 1. Extract Features
    print(f"Loading image from: {image_path}")
    try:
        features, profile, feature_names = extract_features_from_image(image_path, band_indices)
        print(f"[OK] Features extracted: {features.shape}")
        print(f"  Feature names: {feature_names}")
    except Exception as e:
        print(f"[ERROR] Error loading image: {e}")
        return None
    
    # 2. Load Labels
    labels = None
    if os.path.exists(shp_path):
        labels = load_labels_from_shapefile(shp_path, profile)
        unique_labels = np.unique(labels[labels >= 0])
        print(f"[OK] Labels loaded. Unique classes: {unique_labels}")
    else:
        print(f"[WARNING] Shapefile not found: {shp_path}")
        print("  Generating synthetic labels for demonstration...")
        labels = np.random.randint(0, 3, size=features.shape[0])
    
    # Check if we have valid labels
    if labels is None or len(np.unique(labels[labels >= 0])) < 2:
        print("[WARNING] Insufficient label variety. Generating synthetic labels...")
        labels = np.random.randint(0, 3, size=features.shape[0])
    
    # 3. Prepare Data
    X, y = prepare_data(features, labels)
    print(f"[OK] Data prepared: X={X.shape}, y={y.shape}")
    print(f"  Class distribution: {np.bincount(y.astype(int))}")
    
    if X.shape[0] == 0:
        print("[ERROR] No valid data found. Skipping.")
        return None
    
    # 4. Subsample if dataset is too large
    if X.shape[0] > 20000:
        print(f"  Subsampling from {X.shape[0]} to 20000 samples for faster training...")
        indices = np.random.choice(X.shape[0], 20000, replace=False)
        X_train = X[indices]
        y_train = y[indices]
    else:
        X_train = X
        y_train = y
    
    # 5. Train Models
    results = {}
    trained_models = {}
    
    for model_type in model_types:
        print(f"\n  Training {model_type.upper()} model...")
        
        try:
            # Scale data for SVM
            if model_type == 'svm':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                # Split for evaluation
                from sklearn.model_selection import train_test_split
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_train_scaled, y_train, test_size=0.3, random_state=42
                )
                
                clf = train_model(X_tr, y_tr, model_type)
                metrics = evaluate_model(clf, X_te, y_te, 
                                       class_names=['Non-Glacier', 'Glacier', 'Glacier Lake'])
                
                trained_models[model_type] = {'model': clf, 'scaler': scaler}
            else:
                from sklearn.model_selection import train_test_split
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_train, y_train, test_size=0.3, random_state=42
                )
                
                clf = train_model(X_tr, y_tr, model_type)
                metrics = evaluate_model(clf, X_te, y_te,
                                       class_names=['Non-Glacier', 'Glacier', 'Glacier Lake'])
                
                trained_models[model_type] = {'model': clf, 'scaler': None}
            
            results[model_type] = metrics
            print(f"    [OK] {model_type.upper()} Accuracy: {metrics['accuracy']:.4f}")
            
        except Exception as e:
            print(f"    [ERROR] Error training {model_type}: {e}")
            continue
    
    # 6. Generate Classification Maps for Best Model
    if results:
        best_model_type = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        print(f"\n  Best model: {best_model_type.upper()} (Accuracy: {results[best_model_type]['accuracy']:.4f})")
        
        # Full image prediction
        print(f"  Generating full classification map...")
        
        try:
            model_info = trained_models[best_model_type]
            
            if model_info['scaler'] is not None:
                # For SVM: fill NaNs and scale
                feat_mean = np.nanmean(features, axis=0)
                inds = np.where(np.isnan(features))
                features_filled = features.copy()
                features_filled[inds] = np.take(feat_mean, inds[1])
                features_scaled = model_info['scaler'].transform(features_filled)
                all_predictions = model_info['model'].predict(features_scaled)
            else:
                all_predictions = model_info['model'].predict(features)
            
            # Reshape to image dimensions
            height, width = profile['height'], profile['width']
            classification_map = all_predictions.reshape((height, width))
            
            # Save map
            output_dir = r"f:\NIDM-SRM Drone Project Docs\intership"
            output_path = os.path.join(output_dir, f'classification_map_{name}_{best_model_type}.png')
            visualize_classification(
                classification_map, 
                output_path, 
                title=f"GLOF Classification - {name} ({best_model_type.upper()})"
            )
            
            # Save model
            model_path = os.path.join(output_dir, f'trained_model_{name}_{best_model_type}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model_info, f)
            print(f"  [OK] Model saved to {model_path}")
            
        except Exception as e:
            print(f"  [ERROR] Error generating map: {e}")
    
    return {
        'name': name,
        'results': results,
        'models': trained_models,
        'feature_names': feature_names,
        'profile': profile
    }


def main():
    """
    Main training pipeline for all satellite datasets.
    """
    print("\n" + "="*70)
    print(" SATELLITE IMAGE CLASSIFICATION TRAINING PIPELINE")
    print("="*70)
    
    # Configuration
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    shp_path = r"F:\NIDM-SRM Drone Project Docs\GLOF project\image analysis\classified_glacier_dt.shp"
    
    # Define all datasets
    datasets = {
        'Landsat_5': {
            'images_dir': os.path.join(base_dir, 'L5_VRT'),
            'sample_image': 'Landsat5_2009-03-14.vrt',
            'bands': {'Green': 1, 'Red': 2, 'NIR': 3, 'SWIR': 4}
        },
        'Landsat_8_9': {
            'images_dir': os.path.join(base_dir, 'L8_9_VRT-20251212T092651Z-1-001', 'L8_9_VRT'),
            'sample_image': 'Landsat8_2023-02-17.vrt',
            'bands': {'Green': 2, 'Red': 3, 'NIR': 4, 'SWIR': 5}
        },
        'Sentinel_2': {
            'images_dir': os.path.join(base_dir, 'S2_VRT-20251212T092639Z-1-001', 'S2_VRT'),
            'sample_image': 'Sentinel2_2023-01-29.vrt',
            'bands': {'Green': 1, 'Red': 2, 'NIR': 3}
        },
        'LISS_IV': {
            'images_dir': os.path.join(base_dir, 'Liss4Clip_VRT-20251212T092648Z-1-001', 'Liss4Clip_VRT'),
            'sample_image': 'R2F03FEB2025_bands_clipped.vrt',
            'bands': {'Green': 0, 'Red': 1, 'NIR': 2}
        }
    }
    
    all_results = []
    
    # Train on each dataset
    for dataset_name, config in datasets.items():
        image_path = os.path.join(config['images_dir'], config['sample_image'])
        
        result = train_on_satellite_data(
            name=dataset_name,
            image_path=image_path,
            shp_path=shp_path,
            band_indices=config['bands'],
            model_types=['rf', 'svm', 'dt']
        )
        
        if result and result['results']:
            for model_type, metrics in result['results'].items():
                all_results.append({
                    'Dataset': dataset_name,
                    'Model': model_type.upper(),
                    'Accuracy': metrics['accuracy'],
                    'Macro F1': metrics['report']['macro avg']['f1-score'],
                    'Weighted F1': metrics['report']['weighted avg']['f1-score']
                })
    
    # Summary Report
    if all_results:
        print("\n" + "="*70)
        print(" FINAL COMPARISON RESULTS")
        print("="*70)
        
        df = pd.DataFrame(all_results)
        print(df.to_string(index=False))
        
        # Find best performing combination
        best_idx = df['Accuracy'].idxmax()
        best_run = df.loc[best_idx]
        
        print("\n" + "-"*70)
        print(" BEST CLASSIFICATION ACHIEVED BY:")
        print("-"*70)
        print(f"  Dataset:     {best_run['Dataset']}")
        print(f"  Model:       {best_run['Model']}")
        print(f"  Accuracy:    {best_run['Accuracy']:.4f}")
        print(f"  Macro F1:    {best_run['Macro F1']:.4f}")
        print(f"  Weighted F1: {best_run['Weighted F1']:.4f}")
        print("-"*70)
        
        # Save results
        output_path = r"f:\NIDM-SRM Drone Project Docs\intership\training_results_all_satellites.csv"
        df.to_csv(output_path, index=False)
        print(f"\n[OK] Results saved to {output_path}")
    else:
        print("\n[ERROR] No results generated. Please check your data paths.")
    
    print("\n" + "="*70)
    print(" TRAINING PIPELINE COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()





