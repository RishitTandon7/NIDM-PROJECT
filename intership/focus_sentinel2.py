"""
FOCUS: Sentinel-2 Data Processing
This script checks for Real Sentinel-2 Data and processes it to create a perfect map.
"""
import os
import sys
import shutil
import numpy as np
import rasterio
from ml_classification import prepare_data, train_model, evaluate_model
from create_watershed_maps import create_professional_map_with_watersheds
from quick_train import calculate_indices, load_shapefile_labels
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    print("="*80)
    print(" SENTINEL-2 FOCUSED PROCESSING")
    print(" Goal: Create a perfect classification map for December 2021")
    print("="*80)
    
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    s2_dir = os.path.join(base_dir, "S2_VRT-20251212T092639Z-1-001", "S2_VRT")
    shp_path = os.path.join(base_dir, 'GLOF project', 'image analysis', 'classified_glacier_dt.shp')
    output_dir = os.path.join(base_dir, 'intership', 'sentinel2_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Target specific date (Dec 2021) as requested
    target_date = "2021-12-05"
    vrt_name = f"Sentinel2_{target_date}.vrt"
    tif_name = f"Sentinel2_{target_date}.tif"
    
    vrt_path = os.path.join(s2_dir, vrt_name)
    tif_path = os.path.join(s2_dir, tif_name)
    
    print(f"\n1. Checking for data: {target_date}")
    print(f"   Folder: {s2_dir}")
    print(f"   Looking for: {tif_name}")
    
    if not os.path.exists(tif_path):
        print("\n" + "!"*80)
        print(" [ERROR] DATA MISSING!")
        print(f" File not found: {tif_path}")
        print("!"*80)
        print("\nTo fix this:")
        print(f"1. Find your actual Sentinel-2 image file for date {target_date}")
        print(f"2. Rename it to: {tif_name}")
        print(f"3. Copy it into: {s2_dir}")
        print("\nOnce you do this, run this script again to generate the map!")
        print("!"*80)
        
        # Check if VRT exists at least
        if os.path.exists(vrt_path):
            print("\n(The .vrt file exists, but it needs the .tif image to work)")
        return
        
    print(f"   [OK] Found data file: {tif_name}")
    
    # Proceed with processing
    print("\n2. Loading and Processing Data...")
    
    try:
        with rasterio.open(tif_path) as src:
            print(f"   Image size: {src.width} x {src.height}")
            print(f"   Bands: {src.count}")
            
            # Downsample for speed
            scale = min(500/src.width, 500/src.height)
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
            
    except Exception as e:
        print(f"   [ERROR] Could not read image: {e}")
        return

    # Indices and Features
    indices = calculate_indices(data)
    feature_list = [data[i] for i in range(data.shape[0])] + list(indices.values())
    features_stack = np.stack(feature_list, axis=-1)
    features_flat = features_stack.reshape(-1, len(feature_list))
    
    # Labels
    print("\n3. Training Model (Decision Tree)...")
    labels = load_shapefile_labels(shp_path, profile)
    
    if labels is None:
        print("   [ERROR] Could not load labels from shapefile")
        return
        
    X, y = prepare_data(features_flat, labels.flatten())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Use Decision Tree as requested
    clf = train_model(X_train, y_train, 'dt')
    metrics = evaluate_model(clf, X_test, y_test)
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    
    # Generate Map
    print("\n4. Generating Professional Map...")
    predictions = clf.predict(features_flat)
    pred_map = predictions.reshape((new_h, new_w))
    
    map_filename = f"S2_{target_date}_Classification_DT.png"
    map_path = os.path.join(output_dir, map_filename)
    
    create_professional_map_with_watersheds(
        pred_map,
        profile,
        map_path,
        title=f"Sentinel-2 Classification ({target_date})\nDecision Tree Model",
        watershed_shp=shp_path
    )
    
    print("\n" + "="*80)
    print(" SUECCSS!")
    print(f" Map saved to: {map_path}")
    print("="*80)

if __name__ == "__main__":
    main()





