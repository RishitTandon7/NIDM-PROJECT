"""
Batch Training Script - Process All VRT Files
Trains a model and generates a classification map for each date
"""
import os
import numpy as np
import rasterio
from create_aligned_data import create_synthetic_satellite_data
from quick_train import calculate_indices, load_shapefile_labels
from ml_classification import prepare_data, train_model, evaluate_model
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import pandas as pd
from sklearn.model_selection import train_test_split
import fiona
from create_watershed_maps import create_professional_map_with_watersheds


def process_vrt_batch(vrt_dir, dataset_name, shp_path, output_dir, base_dir, max_files=None):
    """
    Process all VRT files in a directory.
    
    Args:
        vrt_dir: Directory containing VRT files
        dataset_name: Name of dataset (e.g., 'Landsat_5')
        shp_path: Path to shapefile with labels
        output_dir: Where to save results
        base_dir: Base directory for data
        max_files: Maximum number of files to process (None = all)
    """
    
    print(f"\n{'='*70}")
    print(f" Processing {dataset_name}")
    print(f"{'='*70}")
    
    # Get VRT files
    vrt_files = sorted([f for f in os.listdir(vrt_dir) if f.endswith('.vrt')])
    
    if max_files:
        vrt_files = vrt_files[:max_files]
    
    print(f"Found {len(vrt_files)} VRT files")
    
    # Get shapefile bounds for creating synthetic data
    with fiona.open(shp_path) as shp:
        bounds = shp.bounds
    
    results = []
    
    for idx, vrt_file in enumerate(vrt_files, 1):
        vrt_path = os.path.join(vrt_dir, vrt_file)
        date = vrt_file.replace('.vrt', '').replace('_clipped', '').replace('_bands', '')
        
        print(f"\n[{idx}/{len(vrt_files)}] {vrt_file}")
        
        try:
            # Try to open VRT
            with rasterio.open(vrt_path) as src:
                # Check if it has actual data
                test_read = src.read(1, window=((0, min(10, src.height)), (0, min(10, src.width))))
                
                if test_read.size == 0 or np.all(test_read == 0):
                    raise ValueError("VRT has no valid data")
                
                # Read with downsampling
                scale = min(300/src.width, 300/src.height)
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
                
                use_real_data = True
                print(f"  [OK] Loaded real VRT data: {new_w}x{new_h}, {data.shape[0]} bands")
                
        except Exception as e:
            # VRT failed, use the pre-created aligned synthetic data
            print(f"  [!] VRT read failed, using pre-generated aligned data")
            
            # Use the aligned data we already created
            aligned_data_dir = os.path.join(base_dir, 'Data_Aligned')
            
            if 'Landsat_5' in dataset_name or 'Landsat_8' in dataset_name:
                aligned_file = os.path.join(aligned_data_dir, 'L8_image_aligned.tif')
            elif 'Sentinel' in dataset_name:
                aligned_file = os.path.join(aligned_data_dir, 'S2_image_aligned.tif')
            else:  # LISS
                aligned_file = os.path.join(aligned_data_dir, 'LISS_image_aligned.tif')
            
            if not os.path.exists(aligned_file):
                print(f"  [X] No aligned data available, skipping")
                continue
            
            with rasterio.open(aligned_file) as src:
                scale = min(300/src.width, 300/src.height)
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
            
            use_real_data = False
            print(f"  [OK] Using aligned synthetic: {new_w}x{new_h}, {data.shape[0]} bands")
        
        # Calculate features
        indices = calculate_indices(data)
        feature_list = [data[i] for i in range(data.shape[0])] + list(indices.values())
        features_stack = np.stack(feature_list, axis=-1)
        features_flat = features_stack.reshape(-1, len(feature_list))
        
        # Load labels
        labels = load_shapefile_labels(shp_path, profile)
        
        if labels is None:
            print("  ✗ Failed to load labels, skipping")
            continue
        
        labels_flat = labels.flatten()
        
        # Prepare data
        X, y = prepare_data(features_flat, labels_flat)
        
        if X.shape[0] < 100:
            print(f"  ✗ Insufficient samples ({X.shape[0]}), skipping")
            continue
        
        # Train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        clf = train_model(X_train, y_train, 'dt')  # Use Decision Tree as requested
        metrics = evaluate_model(clf, X_test, y_test,
                                class_names=['Non-Glacier', 'Glacier', 'Glacier Lake'])
        
        print(f"  [OK] Trained DT: {metrics['accuracy']:.4f} accuracy")
        
        # Generate prediction map
        predictions = clf.predict(features_flat)
        pred_map = predictions.reshape((profile['height'], profile['width']))
        
        # Save map with watershed boundaries
        map_filename = f'{dataset_name}_{date}_classification.png'
        map_path = os.path.join(output_dir, 'maps', map_filename)
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        
        create_professional_map_with_watersheds(
            pred_map,
            profile,
            map_path,
            title=f'{dataset_name} - {date}',
            watershed_shp=shp_path
        )
        
        print(f"  [OK] Map saved: {map_filename}")
        
        # Store results
        results.append({
            'Dataset': dataset_name,
            'Date': date,
            'File': vrt_file,
            'Accuracy': metrics['accuracy'],
            'Macro_F1': metrics['report']['macro avg']['f1-score'],
            'Weighted_F1': metrics['report']['weighted avg']['f1-score'],
            'Data_Source': 'Real VRT' if use_real_data else 'Synthetic'
        })
    
    return results


def main():
    print("\n" + "="*70)
    print(" BATCH TRAINING - ALL VRT DATES")
    print("="*70 + "\n")
    
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    shp_path = os.path.join(base_dir, 'GLOF project', 'image analysis', 'classified_glacier_dt.shp')
    output_dir = os.path.join(base_dir, 'intership')
    
    # VRT directories
    vrt_configs = [
        {
            'dir': os.path.join(base_dir, 'L5_VRT'),
            'name': 'Landsat_5',
            'max_files': 5  # Limit for testing
        },
        {
            'dir': os.path.join(base_dir, 'L8_9_VRT-20251212T092651Z-1-001', 'L8_9_VRT'),
            'name': 'Landsat_8_9',
            'max_files': 5
        },
        {
            'dir': os.path.join(base_dir, 'S2_VRT-20251212T092639Z-1-001', 'S2_VRT'),
            'name': 'Sentinel_2',
            'max_files': 5
        },
        {
            'dir': os.path.join(base_dir, 'Liss4Clip_VRT-20251212T092648Z-1-001', 'Liss4Clip_VRT'),
            'name': 'LISS_IV',
            'max_files': 5
        }
    ]
    
    all_results = []
    
    for config in vrt_configs:
        if not os.path.exists(config['dir']):
            print(f"\n[WARNING] Directory not found: {config['dir']}")
            continue
        
        results = process_vrt_batch(
            config['dir'],
            config['name'],
            shp_path,
            output_dir,
            base_dir,
            max_files=config['max_files']
        )
        
        all_results.extend(results)
    
    # Save summary
    if all_results:
        df = pd.DataFrame(all_results)
        
        print("\n" + "="*70)
        print(" RESULTS SUMMARY")
        print("="*70 + "\n")
        print(df[['Dataset', 'Date', 'Accuracy', 'Data_Source']].to_string(index=False))
        
        csv_path = os.path.join(output_dir, 'batch_training_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n[OK] Results saved: {csv_path}")
    
    print("\n" + "="*70)
    print(" BATCH TRAINING COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Import PIL for downsampling
    try:
        from PIL import Image
    except ImportError:
        print("[WARNING] PIL not available, downsampling may be limited")
    
    main()





