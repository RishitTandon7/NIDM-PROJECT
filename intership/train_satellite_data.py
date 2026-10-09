"""
Flexible Satellite Image Training Script
Handles both VRT files and regular TIFF files
"""
import os
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import fiona
from satellite_processing import extract_features_from_image
from ml_classification import prepare_data, train_model, evaluate_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pickle


def load_labels_from_shapefile(shp_path, reference_profile):
    """
    Rasterize a shapefile to match the reference image profile.
    """
    if not os.path.exists(shp_path):
        print(f"[WARNING] Shapefile not found: {shp_path}")
        return None
        
    print(f"Rasterizing shapefile...")
    
    try:
        with fiona.open(shp_path, "r") as shapefile:
            shapes = [
                (feature["geometry"], feature["properties"]["class_id"])
                for feature in shapefile
            ]
        
        image_shape = (reference_profile['height'], reference_profile['width'])
        transform = reference_profile['transform']
        
        labels = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=image_shape,
            transform=transform,
            fill=-1,
            dtype='int32'
        )
        
        return labels.flatten()
    except Exception as e:
        print(f"[ERROR] Failed to rasterize shapefile: {e}")
        return None


def visualize_classification(classification_map, output_path, title="Classification Map"):
    """Create and save classification map visualization."""
    plt.figure(figsize=(12, 10))
    
    cmap = ListedColormap(['lightgreen', 'deepskyblue', 'red'])
    
    img = plt.imshow(classification_map, cmap=cmap, interpolation='nearest')
    cbar = plt.colorbar(img, ticks=[0, 1, 2], label='Class', shrink=0.8)
    cbar.set_ticklabels(['Non-Glacier', 'Glacier', 'Glacier Lake'])
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Map saved: {os.path.basename(output_path)}")


def train_on_image(name, image_path, shp_path, band_indices):
    """
    Train models on a single satellite image.
    """
    print(f"\n{'='*70}")
    print(f" {name}")
    print(f"{'='*70}")
    
    # 1. Check if image exists
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return None
    
    # 2. Extract Features
    print(f"Loading: {os.path.basename(image_path)}")
    try:
        features, profile, feature_names = extract_features_from_image(image_path, band_indices)
        print(f"[OK] Extracted features: {features.shape}")
        print(f"     Features: {', '.join(feature_names)}")
    except Exception as e:
        print(f"[ERROR] Failed to load image: {str(e)[:100]}")
        return None
    
    # 3. Load Labels
    labels = None
    if os.path.exists(shp_path):
        labels = load_labels_from_shapefile(shp_path, profile)
        if labels is not None:
            unique_labels = np.unique(labels[labels >= 0])
            print(f"[OK] Labels loaded. Classes: {unique_labels}")
    
    # Generate synthetic labels if needed
    if labels is None or len(np.unique(labels[labels >= 0])) < 2:
        print("[WARNING] Using synthetic labels for demonstration")
        labels = np.random.randint(0, 3, size=features.shape[0])
    
    # 4. Prepare Data
    X, y = prepare_data(features, labels)
    print(f"[OK] Prepared: {X.shape[0]} valid samples")
    
    class_counts = np.bincount(y.astype(int))
    for i, count in enumerate(class_counts):
        print(f"     Class {i}: {count} samples ({count/len(y)*100:.1f}%)")
    
    if X.shape[0] < 100:
        print("[ERROR] Insufficient samples. Skipping.")
        return None
    
    # 5. Subsample if too large
    if X.shape[0] > 15000:
        print(f"     Subsampling to 15000 samples...")
        indices = np.random.choice(X.shape[0], 15000, replace=False)
        X_sample, y_sample = X[indices], y[indices]
    else:
        X_sample, y_sample = X, y
    
    # 6. Train Models
    model_types = ['rf', 'svm', 'dt']
    results = {}
    trained_models = {}
    
    for model_type in model_types:
        print(f"\n  Training {model_type.upper()}...")
        
        try:
            # Prepare training/test split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sample, y_sample, test_size=0.3, random_state=42, stratify=y_sample
            )
            
            # Scale data for SVM
            if model_type == 'svm':
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_te_scaled = scaler.transform(X_te)
                
                clf = train_model(X_tr_scaled, y_tr, model_type)
                metrics = evaluate_model(clf, X_te_scaled, y_te, 
                                       class_names=['Non-Glacier', 'Glacier', 'Glacier Lake'])
                trained_models[model_type] = {'model': clf, 'scaler': scaler}
            else:
                clf = train_model(X_tr, y_tr, model_type)
                metrics = evaluate_model(clf, X_te, y_te,
                                       class_names=['Non-Glacier', 'Glacier', 'Glacier Lake'])
                trained_models[model_type] = {'model': clf, 'scaler': None}
            
            results[model_type] = metrics
            print(f"    Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['report']['weighted avg']['f1-score']:.4f}")
            
        except Exception as e:
            print(f"    [ERROR] Training failed: {str(e)[:80]}")
            continue
    
    # 7. Generate Map for Best Model
    if not results:
        print("[ERROR] No models trained successfully")
        return None
    
    best_model_type = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\n  [BEST] {best_model_type.upper()} - Accuracy: {results[best_model_type]['accuracy']:.4f}")
    
    # Create classification map
    try:
        model_info = trained_models[best_model_type]
        
        # Handle NaN values and prepare features
        if model_info['scaler'] is not None:
            feat_mean = np.nanmean(features, axis=0)
            features_filled = features.copy()
            nan_mask = np.isnan(features_filled)
            for col in range(features_filled.shape[1]):
                features_filled[nan_mask[:, col], col] = feat_mean[col]
            features_pred = model_info['scaler'].transform(features_filled)
        else:
            features_pred = features
        
        predictions = model_info['model'].predict(features_pred)
        
        # Reshape and save
        height, width = profile['height'], profile['width']
        class_map = predictions.reshape((height, width))
        
        output_dir = r"f:\NIDM-SRM Drone Project Docs\intership"
        map_path = os.path.join(output_dir, f'map_{name}_{best_model_type}.png')
        visualize_classification(class_map, map_path, 
                               title=f"GLOF Classification - {name} ({best_model_type.upper()})")
        
        # Save model
        model_path = os.path.join(output_dir, f'model_{name}_{best_model_type}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"[OK] Model saved: {os.path.basename(model_path)}")
        
    except Exception as e:
        print(f"[ERROR] Map generation failed: {str(e)[:80]}")
    
    return {
        'name': name,
        'results': results,
        'models': trained_models,
        'feature_names': feature_names
    }


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print(" SATELLITE IMAGE CLASSIFICATION - TRAINING PIPELINE")
    print("="*70 + "\n")
    
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    shp_path = os.path.join(base_dir, r"GLOF project\image analysis\classified_glacier_dt.shp")
    
    # Define datasets - using available TIFF files from Data folder
    datasets = {
        'Landsat_8_Data': {
            'path': os.path.join(base_dir, 'Data', 'L8_image.tif'),
            'bands': {'Green': 1, 'Red': 2, 'NIR': 3, 'SWIR': 4}
        },
        'Sentinel_2_Data': {
            'path': os.path.join(base_dir, 'Data', 'S2_image.tif'),
            'bands': {'Green': 0, 'Red': 1, 'NIR': 2}
        },
        'LISS_IV_Data': {
            'path': os.path.join(base_dir, 'Data', 'LISS_image.tif'),
            'bands': {'Green': 0, 'Red': 1, 'NIR': 2}
        }
    }
    
    all_results = []
    
    # Train on each dataset
    for dataset_name, config in datasets.items():
        result = train_on_image(
            name=dataset_name,
            image_path=config['path'],
            shp_path=shp_path,
            band_indices=config['bands']
        )
        
        if result and result['results']:
            for model_type, metrics in result['results'].items():
                all_results.append({
                    'Dataset': dataset_name,
                    'Model': model_type.upper(),
                    'Accuracy': metrics['accuracy'],
                    'Macro F1': metrics['report']['macro avg']['f1-score'],
                    'Weighted F1': metrics['report']['weighted avg']['f1-score'],
                    'Precision': metrics['report']['weighted avg']['precision'],
                    'Recall': metrics['report']['weighted avg']['recall']
                })
    
    # Summary Report
    if all_results:
        print("\n" + "="*70)
        print(" FINAL COMPARISON RESULTS")
        print("="*70 + "\n")
        
        df = pd.DataFrame(all_results)
        print(df.to_string(index=False))
        
        # Best model
        best_idx = df['Accuracy'].idxmax()
        best = df.loc[best_idx]
        
        print("\n" + "-"*70)
        print(" BEST MODEL:")
        print("-"*70)
        print(f"  Dataset:      {best['Dataset']}")
        print(f"  Model:        {best['Model']}")
        print(f"  Accuracy:     {best['Accuracy']:.4f}")
        print(f"  Weighted F1:  {best['Weighted F1']:.4f}")
        print(f"  Precision:    {best['Precision']:.4f}")
        print(f"  Recall:       {best['Recall']:.4f}")
        print("-"*70 + "\n")
        
        # Save results
        output_path = os.path.join(base_dir, 'intership', 'satellite_training_results.csv')
        df.to_csv(output_path, index=False)
        print(f"[OK] Results saved to: {os.path.basename(output_path)}\n")
    else:
        print("\n[ERROR] No results generated.\n")
    
    print("="*70)
    print(" TRAINING COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()





