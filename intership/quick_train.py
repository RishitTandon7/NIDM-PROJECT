"""
Optimized Training Script - Based on Existing Workflow
Fast training with downsampled data matching reference notebook approach
"""
import os
import numpy as np
import rasterio
from satellite_processing import extract_features_from_image
from ml_classification import prepare_data, train_model, evaluate_model
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import fiona
from rasterio.features import rasterize
from sklearn.model_selection import train_test_split
import pandas as pd


def load_and_downsample_image(img_path, target_size=300):
    """Load image and downsample to target size for faster processing."""
    with rasterio.open(img_path) as src:
        # Calculate downsample factor
        scale = min(target_size / src.width, target_size / src.height)
        
        # Calculate new dimensions
        new_width = int(src.width * scale)
        new_height = int(src.height * scale)
        
        # Read with downsampling
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=rasterio.enums.Resampling.bilinear
        )
        
        # Update profile
        profile = src.profile.copy()
        profile.update({
            'width': new_width,
            'height': new_height,
            'transform': src.transform * src.transform.scale(
                (src.width / new_width),
                (src.height / new_height)
            )
        })
        
        return data, profile


def load_shapefile_labels(shp_path, profile):
    """Load and rasterize shapefile labels."""
    if not os.path.exists(shp_path):
        print(f"[WARNING] Shapefile not found: {shp_path}")
        return None
    
    try:
        with fiona.open(shp_path) as shp:
            shapes = [
                (feature["geometry"], feature["properties"]["class_id"])
                for feature in shp
            ]
        
        labels = rasterize(
            shapes=shapes,
            out_shape=(profile['height'], profile['width']),
            transform=profile['transform'],
            fill=-1,
            dtype='int32'
        )
        
        return labels
    except Exception as e:
        print(f"[ERROR] Failed to load labels: {e}")
        return None


def calculate_indices(data):
    """Calculate spectral indices from bands."""
    indices = {}
    
    # Assume standard band order: Green=1, Red=2, NIR=3, SWIR=4 (0-indexed)
    if data.shape[0] >= 3:
        green = data[1].astype(float)
        red = data[2].astype(float)
        nir = data[3].astype(float) if data.shape[0] >= 4 else data[2].astype(float)
        
        # NDVI
        ndvi = np.divide(nir - red, nir + red, 
                        out=np.zeros_like(nir), 
                        where=(nir + red) != 0)
        indices['NDVI'] = ndvi
        
        # NDWI
        ndwi = np.divide(green - nir, green + nir,
                        out=np.zeros_like(green),
                        where=(green + nir) != 0)
        indices['NDWI'] = ndwi
        
        # If SWIR available
        if data.shape[0] >= 5:
            swir = data[4].astype(float)
            # MNDWI
            mndwi = np.divide(green - swir, green + swir,
                            out=np.zeros_like(green),
                            where=(green + swir) != 0)
            indices['MNDWI'] = mndwi
    
    return indices


def quick_train(name, img_path, shp_path, output_dir):
    """Quick training with downsampled data."""
    
    print(f"\n{'='*70}")
    print(f"Training: {name}")
    print(f"{'='*70}")
    
    # Load and downsample image
    print("Loading and downsampling image...")
    data, profile = load_and_downsample_image(img_path, target_size=300)
    
    print(f"  Downsampled to: {profile['width']} x {profile['height']}")
    print(f"  Bands: {data.shape[0]}")
    
    # Calculate indices
    indices = calculate_indices(data)
    
    # Stack features
    feature_list = [data[i] for i in range(data.shape[0])] + list(indices.values())
    features_stack = np.stack(feature_list, axis=-1)
    features_flat = features_stack.reshape(-1, len(feature_list))
    
    print(f"  Features shape: {features_flat.shape}")
    
    # Load labels
    print("Loading labels...")
    labels = load_shapefile_labels(shp_path, profile)
    
    if labels is not None:
        labels_flat = labels.flatten()
        unique_vals = np.unique(labels_flat[labels_flat >= 0])
        print(f"  Found {len(unique_vals)} classes: {unique_vals}")
    else:
        print("  No valid labels - using synthetic for demo")
        labels_flat = np.random.randint(0, 3, size=features_flat.shape[0])
    
    # Prepare data
    X, y = prepare_data(features_flat, labels_flat)
    print(f"  Valid samples: {X.shape[0]}")
    
    if X.shape[0] < 100:
        print("[WARNING] Insufficient valid pixels")
        return None
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train models
    results = {}
    models = {}
    
    for model_type in ['rf', 'dt']:  # Skip SVM for speed
        print(f"\n  Training {model_type.upper()}...")
        
        clf = train_model(X_train, y_train, model_type)
        metrics = evaluate_model(clf, X_test, y_test,
                                class_names=['Non-Glacier', 'Glacier', 'Glacier Lake'])
        
        results[model_type]  = metrics
        models[model_type] = clf
        
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
    
    # Generate map with best model
    best_type = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\n  Best: {best_type.upper()} ({results[best_type]['accuracy']:.4f})")
    
    # Predict on full image
    predictions = models[best_type].predict(features_flat)
    pred_map = predictions.reshape((profile['height'], profile['width']))
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['#90EE90', '#1E90FF', '#FF6347']
    cmap = ListedColormap(colors)
    
    # Get extent
    transform = profile['transform']
    left = transform[2]
    top = transform[5]
    right = left + profile['width'] * transform[0]
    bottom = top + profile['height'] * transform[4]
    extent = [left, right, bottom, top]
    
    ax.imshow(pred_map, cmap=cmap, extent=extent, aspect='auto', interpolation='nearest')
    ax.set_xlabel('Longitude', fontweight='bold')
    ax.set_ylabel('Latitude', fontweight='bold')
    ax.set_title(f'GLOF Classification - {name} ({best_type.upper()})', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='black', label='Non-Glacier'),
        Patch(facecolor=colors[1], edgecolor='black', label='Glacier'),
        Patch(facecolor=colors[2], edgecolor='black', label='Glacier Lake')
    ]
    ax.legend(handles=legend_elements, loc='lower left', frameon=True, shadow=True)
    
    plt.tight_layout()
    map_path = os.path.join(output_dir, f'classification_{name}_{best_type}.png')
    plt.savefig(map_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n  [OK] Map saved: {os.path.basename(map_path)}")
    
    return {
        'name': name,
        'results': results,
        'best_model': best_type
    }


def main():
    print("\n" + "="*70)
    print(" OPTIMIZED SATELLITE CLASSIFICATION TRAINING")
    print(" (Based on Reference Workflow)")
    print("="*70 + "\n")
    
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    shp_path = os.path.join(base_dir, 'GLOF project', 'image analysis', 'classified_glacier_dt.shp')
    output_dir = os.path.join(base_dir, 'intership')
    
    datasets = {
        'Landsat_8': os.path.join(base_dir, 'Data_Aligned', 'L8_image_aligned.tif'),
        'Sentinel_2': os.path.join(base_dir, 'Data_Aligned', 'S2_image_aligned.tif'),
        'LISS_IV': os.path.join(base_dir, 'Data_Aligned', 'LISS_image_aligned.tif')
    }
    
    all_results = []
    
    for name, img_path in datasets.items():
        if not os.path.exists(img_path):
            print(f"[WARNING] Skipping {name} - file not found")
            continue
        
        result = quick_train(name, img_path, shp_path, output_dir)
        
        if result:
            for model_type, metrics in result['results'].items():
                all_results.append({
                    'Dataset': name,
                    'Model': model_type.upper(),
                    'Accuracy': metrics['accuracy'],
                    'Macro F1': metrics['report']['macro avg']['f1-score'],
                    'Weighted F1': metrics['report']['weighted avg']['f1-score']
                })
    
    # Summary
    if all_results:
        print("\n" + "="*70)
        print(" RESULTS SUMMARY")
        print("="*70 + "\n")
        
        df = pd.DataFrame(all_results)
        print(df.to_string(index=False))
        
        # Save
        csv_path = os.path.join(output_dir, 'quick_training_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n[OK] Results saved: {os.path.basename(csv_path)}")
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()





