"""
Training Script - Using Existing Clipped LISS-IV VRTs
Following faculty guidance to use the clipped VRT files from Liss4Clip_VRT folder
"""
import os
import numpy as np
import rasterio
from quick_train import calculate_indices, load_shapefile_labels
from ml_classification import prepare_data, train_model, evaluate_model
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import pandas as pd


def main():
    print("\n" + "="*70)
    print(" LISS-IV TRAINING - Using Clipped VRT Files")
    print("="*70 + "\n")
    
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    
    # Use the clipped VRT directory
    vrt_dir = os.path.join(base_dir, 'Liss4Clip_VRT-20251212T092648Z-1-001', 'Liss4Clip_VRT')
    shp_path = os.path.join(base_dir, 'GLOF project', 'image analysis', 'classified_glacier_dt.shp')
    output_dir = os.path.join(base_dir, 'intership')
    
    # Get one sample VRT to train on
    vrt_files = [f for f in os.listdir(vrt_dir) if f.endswith('_clipped.vrt')]
    print(f"Found {len(vrt_files)} clipped VRT files")
    
    # Use a recent file
    sample_vrt = os.path.join(vrt_dir, 'R2F03FEB2025_bands_clipped.vrt')
    
    if not os.path.exists(sample_vrt):
        # Use first available
        sample_vrt = os.path.join(vrt_dir, vrt_files[0])
    
    print(f"\nTraining on: {os.path.basename(sample_vrt)}")
    
    # Load VRT
    with rasterio.open(sample_vrt) as src:
        print(f"  Size: {src.width} x {src.height}")
        print(f"  Bands: {src.count}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
        
        # Read data (downsample if too large)
        if src.width > 1000 or src.height > 1000:
            scale = min(800/src.width, 800/src.height)
            new_w = int(src.width * scale)
            new_h = int(src.height * scale)
            
            data = src.read(
                out_shape=(src.count, new_h, new_w),
                resampling=rasterio.enums.Resampling.bilinear
            )
            
            profile = src.profile.copy()
            profile.update({
                'width': new_w,
                'height': new_h,
                'transform': src.transform * src.transform.scale(
                    (src.width / new_w),
                    (src.height / new_h)
                )
            })
            print(f"  Downsampled to: {new_w} x {new_h}")
        else:
            data = src.read()
            profile = src.profile.copy()
    
    # Calculate indices
    indices = calculate_indices(data)
    
    # Stack features
    feature_list = [data[i] for i in range(data.shape[0])] + list(indices.values())
    features_stack = np.stack(feature_list, axis=-1)
    features_flat = features_stack.reshape(-1, len(feature_list))
    
    print(f"  Features: {features_flat.shape}")
    
    # Load labels
    print("\nLoading labels from shapefile...")
    labels = load_shapefile_labels(shp_path, profile)
    
    if labels is not None:
        labels_flat = labels.flatten()
        unique = np.unique(labels_flat[labels_flat >= 0])
        print(f"  Classes found: {unique}")
        
        # Prepare data
        X, y = prepare_data(features_flat, labels_flat)
        print(f"  Valid samples: {X.shape[0]}")
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train models
        print("\nTraining models...")
        results = {}
        
        for model_type in ['rf', 'dt']:
            print(f"  {model_type.upper()}...", end=" ")
            clf = train_model(X_train, y_train, model_type)
            metrics = evaluate_model(clf, X_test, y_test,
                                    class_names=['Non-Glacier', 'Glacier', 'Glacier Lake'])
            results[model_type] = (clf, metrics)
            print(f"Accuracy: {metrics['accuracy']:.4f}")
        
        # Best model
        best_type = max(results.items(), key=lambda x: x[1][1]['accuracy'])[0]
        best_clf, best_metrics = results[best_type]
        
        print(f"\n  Best: {best_type.upper()} - {best_metrics['accuracy']:.4f}")
        
        # Generate map
        predictions = best_clf.predict(features_flat)
        pred_map = predictions.reshape((profile['height'], profile['width']))
        
        # Visualize
        fig, ax = plt.subplots(figsize=(14, 11))
        
        colors = ['#90EE90', '#1E90FF', '#FF6347']
        cmap = ListedColormap(colors)
        
        transform = profile['transform']
        left = transform[2]
        top = transform[5]
        right = left + profile['width'] * transform[0]
        bottom = top + profile['height'] * transform[4]
        extent = [left, right, bottom, top]
        
        ax.imshow(pred_map, cmap=cmap, extent=extent, aspect='auto', interpolation='nearest')
        ax.set_xlabel('Longitude', fontweight='bold', fontsize=12)
        ax.set_ylabel('Latitude', fontweight='bold', fontsize=12)
        ax.set_title(f'GLOF Classification - LISS-IV ({best_type.upper()})', 
                    fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        legend_elements = [
            Patch(facecolor=colors[0], edgecolor='black', label='Non-Glacier'),
            Patch(facecolor=colors[1], edgecolor='black', label='Glacier'),
            Patch(facecolor=colors[2], edgecolor='black', label='Glacier Lake')
        ]
        ax.legend(handles=legend_elements, loc='lower left', frameon=True, shadow=True, fontsize=11)
        
        plt.tight_layout()
        map_path = os.path.join(output_dir, f'LISS4_classification_{best_type}.png')
        plt.savefig(map_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"\n[OK] Map saved: {os.path.basename(map_path)}")
        
        # Save results
        df = pd.DataFrame([{
            'Dataset': 'LISS-IV (Clipped VRT)',
            'File': os.path.basename(sample_vrt),
            'Model': best_type.upper(),
            'Accuracy': best_metrics['accuracy'],
            'Macro F1': best_metrics['report']['macro avg']['f1-score'],
            'Weighted F1': best_metrics['report']['weighted avg']['f1-score']
        }])
        
        csv_path = os.path.join(output_dir, 'liss4_vrt_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"[OK] Results saved: {os.path.basename(csv_path)}")
        
    else:
        print("[ERROR] Could not load labels")
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()





