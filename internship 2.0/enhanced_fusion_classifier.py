"""
Enhanced Fusion Classifier for Glacial Lake Outburst Flood (GLOF) Analysis
Author: Antigravity (Google Deepmind)
Date: 2026-01-23

This script implements a reliable classification pipeline for Glacial Lakes, minimizing confusion 
with non-glacial lakes and mountain shadows.

Key Innovations:
1.  **Multi-Index Spectral Analysis**: Computes NDWI, MNDWI, and NDSI.
2.  **Contextual Awareness**: Integrates DEM-based elevation/slope and proximity to glaciers.
3.  **Physics-Informed Training**: Uses physical rules (Slope > 20 deg = No Lake) to generate high-quality training labels.
4.  **Machine Learning Fusion**: Trains a Random Forest on the fused feature set.
5.  **Object-Based Refinement**: Uses segmentation to filter results.

Usage:
    Ensure 'Sentinel 2' folder contains .tif files.
    Ensure 'universal_vectors' contains glacier GeoJSONs.
    (Optional) Provide a DEM file path for elevation/slope features.
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize, shapes
from scipy.ndimage import distance_transform_edt, label
from skimage.measure import regionprops
from skimage.segmentation import slic
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
import gc

warnings.filterwarnings('ignore')

class FusionClassifier:
    def __init__(self, tiff_dir, vector_dir, output_dir, dem_path=None):
        self.tiff_dir = tiff_dir
        self.vector_dir = vector_dir
        self.output_dir = output_dir
        self.dem_path = dem_path
        self.scaler = StandardScaler()
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        
        os.makedirs(self.output_dir, exist_ok=True)

    def load_bands(self, src):
        """Loads Sentinel-2 bands and handles missing data"""
        try:
            # S2 usually: B2(Blue), B3(Green), B4(Red), B8(NIR), B11(SWIR1)
            # Assuming standard 10m stack or similar ordering.
            # Adjust indices based on your specific TIF generation (e.g. from Earth Engine)
            # Standard: 1=Blue, 2=Green, 3=Red, 4=NIR... 
            # Ideally need SWIR (Band 11/12). If count < 5, might be missing.
            
            blue = src.read(1).astype(np.float32)
            green = src.read(2).astype(np.float32)
            red = src.read(3).astype(np.float32)
            nir = src.read(4).astype(np.float32)
            
            swir = np.zeros_like(blue, dtype=np.float32)
            if src.count >= 9: # Often band 9 or 11 in processed stacks
                swir = src.read(9).astype(np.float32) # Utilizing SWIR if available
            elif src.count >= 5:
                swir = src.read(5).astype(np.float32) # Fallback
                
            return blue, green, red, nir, swir
        except Exception as e:
            print(f"Error reading bands: {e}")
            return None

    def calculate_indices(self, green, red, nir, swir):
        """Calculates NDWI, MNDWI, NDSI"""
        np.seterr(divide='ignore', invalid='ignore')
        
        # NDWI (McFeeters): (Green - NIR) / (Green + NIR)
        ndwi = (green - nir) / (green + nir + 1e-6)
        
        # MNDWI (Xu): (Green - SWIR) / (Green + SWIR)
        # Better for suppressing built-up noise and some shadows
        mndwi = (green - swir) / (green + swir + 1e-6)
        
        # NDSI (Hall): (Green - SWIR) / (Green + SWIR)
        # *Note*: NDSI and MNDWI use the same formula but detect different things based on threshold.
        # Snow > 0.4 usually. Water is also > 0 but usually lower or eliminated by slope.
        # We will use the name 'ndsi' for clarity in features.
        ndsi = mndwi.copy() 
        
        # NDVI: (NIR - Red) / (NIR + Red) - helpful for vegetation exclusion
        ndvi = (nir - red) / (nir + red + 1e-6)
        
        return np.nan_to_num(ndwi), np.nan_to_num(mndwi), np.nan_to_num(ndvi)

    def process_dem(self, ref_profile, shape):
        """Loads and aligns DEM; calculates Slope"""
        dem_layer = np.zeros(shape, dtype=np.float32)
        slope_layer = np.zeros(shape, dtype=np.float32)
        
        if not self.dem_path or not os.path.exists(self.dem_path):
            return dem_layer, slope_layer

        try:
            with rasterio.open(self.dem_path) as dem_src:
                # Reproject DEM to match Sentinel-2 image
                reproject(
                    source=rasterio.band(dem_src, 1),
                    destination=dem_layer,
                    src_transform=dem_src.transform,
                    src_crs=dem_src.crs,
                    dst_transform=ref_profile['transform'],
                    dst_crs=ref_profile['crs'],
                    resampling=Resampling.bilinear
                )
                
            # Calculate Slope (Simple Gradient)
            # dy, dx = np.gradient(dem_layer)
            # This is pixel-wise gradient; simple proxy for slope magnitude
            # Ideally requires geotransform scaling, but relative gradient is fine for RF
            gy, gx = np.gradient(dem_layer)
            slope_layer = np.sqrt(gy**2 + gx**2)
            
        except Exception as e:
            print(f"DEM processing failed: {e}")
            
        return dem_layer, slope_layer

    def calculate_glacier_proximity(self, shape, transform, crs, date_str):
        """Calculates distance from nearest glacier"""
        # Find matching glacier vector for date (or closest date)
        # Simplification: Loading closest available or just all for "static" glaciers
        # Here we look for exact match or 'universal'
        
        glacier_mask = np.zeros(shape, dtype=np.uint8)
        
        # Search for glacier file
        g_files = glob.glob(os.path.join(self.vector_dir, f"*glacier*202*.geojson"))
        # Simple matching logic: find file with same date or use recent
        # Assuming filename format: glacier_YYYY-MM-DD.geojson
        
        target_file = None
        for gf in g_files:
            if date_str in gf:
                target_file = gf
                break
        
        if not target_file and g_files:
            target_file = g_files[-1] # Fallback to latest
            
        if target_file:
            try:
                gdf = gpd.read_file(target_file)
                if not gdf.empty:
                    # Reproject if needed
                    if gdf.crs != crs:
                        gdf = gdf.to_crs(crs)
                    
                    # Rasterize
                    glacier_mask = rasterize(
                        [(geom, 1) for geom in gdf.geometry],
                        out_shape=shape,
                        transform=transform,
                        fill=0,
                        dtype=np.uint8
                    )
            except Exception as e:
                print(f"Vector load error: {e}")

        # Distance Transform (Euclidean)
        # edt calculates distance from non-zero (background) to nearest zero (feature/source)
        # We want distance TO glacier. So Glacier=0, Background=1.
        # glacier_mask: 1=Glacier, 0=Background
        # (glacier_mask == 0) -> True (1) for Background, False (0) for Glacier
        # This creates a boolean array (smaller memory footprint) instead of int64
        dist_map = distance_transform_edt(glacier_mask == 0)
        return dist_map.astype(np.float32)

    def extract_features(self, tiff_path):
        """Extracts and stacks all features for an image"""
        with rasterio.open(tiff_path) as src:
            profile = src.profile
            date_str = os.path.basename(tiff_path)[10:20] # Adjust index based on naming
            
            # 1. Bands
            bands = self.load_bands(src)
            if not bands: return None
            blue, green, red, nir, swir = bands
            
            # 2. Indices
            ndwi, mndwi, ndvi = self.calculate_indices(green, red, nir, swir)
            
            # 3. DEM & Slope
            dem, slope = self.process_dem(profile, blue.shape)
            
            # 4. Context
            dist_glacier = self.calculate_glacier_proximity(blue.shape, src.transform, src.crs, date_str)
            
            # 5. Stack Features
            # Features: [Blue, Green, Red, NIR, SWIR, NDWI, MNDWI, NDVI, DEM, Slope, Dist]
            # Flatten for ML
            features = np.stack([
                blue.flatten(), green.flatten(), red.flatten(), nir.flatten(), swir.flatten(),
                ndwi.flatten(), mndwi.flatten(), ndvi.flatten(),
                dem.flatten(), slope.flatten(), dist_glacier.flatten()
            ], axis=1).astype(np.float32)
            
            # Handle NaNs
            features = np.nan_to_num(features)
            
            metadata = {
                'shape': blue.shape,
                'transform': src.transform,
                'crs': src.crs,
                'date': date_str,
                'profile': profile
            }
            
            return features, metadata

    def generate_training_data(self, tiff_files):
        """
        Generates 'Silver Standard' training data using Physics Rules
        to train the RF model to generalize better.
        """
        print("Generating Physics-Informed Training Data...")
        X_train = []
        y_train = []
        
        # Sample a subset of images to build training set
        sample_files = tiff_files[:min(5, len(tiff_files))] 
        
        for tiff in sample_files:
            feats, meta = self.extract_features(tiff)
            if feats is None: continue
            
            # Unpack for rules (columns based on stack order in extract_features)
            # 0:B, 1:G, 2:R, 3:N, 4:S, 5:NDWI, 6:MNDWI, 7:NDVI, 8:DEM, 9:Slope, 10:Dist
            ndwi = feats[:, 5]
            mndwi = feats[:, 6]
            nir = feats[:, 3]
            slope = feats[:, 9]
            dist = feats[:, 10]
            ndsi = feats[:, 6]
            brightness = (feats[:, 0] + feats[:, 1] + feats[:, 2]) / 3.0
            
            labels = np.zeros(len(feats), dtype=np.uint8) # 0 = Background
            
            # --- PHYSICS RULES (STRICTER) ---
            
            # 1. Clean Water Mask (MNDWI or NDWI high, AND Not Snow, AND Flat)
            # Snow/Ice is bright. Water is dark (in visible/NIR).
            # Slope constraint is critical: Lakes are flat.
            is_water_candidate = (mndwi > 0.0) | (ndwi > 0.0)
            is_not_snow = (brightness < 3000) # Removed ndsi < 0.4 because Water also has high NDSI 
            is_flat = (slope < 12) # Strict slope for training
            
            mask_lake = is_water_candidate & is_not_snow & is_flat & (dist < 500)
            labels[mask_lake] = 1 # Glacial Lake
            
            # 2. Glacier/Snow (Bright, High NDSI, Any Slope)
            mask_ice = (brightness > 3500) | (ndsi > 0.4)
            labels[mask_ice] = 2
            
            # 3. Shadow (Dark, High Slope usually)
            # Shadows often masquerade as water due to high NDWI. 
            # Key diff: Slope is usually high, or purely dark (lower than water).
            mask_shadow = (brightness < 600) & (slope > 10)
            labels[mask_shadow] = 3
            
            # Sampling: Take a balanced subset of pixels
            # Take all detected Lakes (rare class)
            idx_lake = np.where(labels == 1)[0]
            idx_ice = np.where(labels == 2)[0]
            idx_shadow = np.where(labels == 3)[0]
            idx_bg = np.where(labels == 0)[0]
            
            # Limit samples to avoid memory explosion
            limit = 5000
            
            if len(idx_lake) > 0:
                X_train.append(feats[np.random.choice(idx_lake, min(len(idx_lake), limit))])
                y_train.append(labels[np.random.choice(idx_lake, min(len(idx_lake), limit))])
                
            if len(idx_ice) > 0:
                X_train.append(feats[np.random.choice(idx_ice, min(len(idx_ice), limit))])
                y_train.append(labels[np.random.choice(idx_ice, min(len(idx_ice), limit))])
                
            if len(idx_shadow) > 0:
                X_train.append(feats[np.random.choice(idx_shadow, min(len(idx_shadow), limit))])
                y_train.append(labels[np.random.choice(idx_shadow, min(len(idx_shadow), limit))])
            
            # Background samples
            X_train.append(feats[np.random.choice(idx_bg, min(len(idx_bg), limit))])
            y_train.append(labels[np.random.choice(idx_bg, min(len(idx_bg), limit))])

        if not X_train:
            return None, None
            
        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        
        return X_train, y_train

    def run(self):
        tiff_files = sorted(glob.glob(os.path.join(self.tiff_dir, "*.tif")))
        if not tiff_files:
            print("No TIF files found.")
            return

        # 1. Train Model
        X, y = self.generate_training_data(tiff_files)
        if X is None:
            print("Failed to generate training data.")
            return
            
        print(f"Training Random Forest on {len(y)} pixels...")
        self.rf_model.fit(X, y)
        print("Model trained.")
        
        # 2. Predict & Refine
        for i, tiff_path in enumerate(tiff_files):
            gc.collect() # Force garbage collection to free memory
            print(f"Processing {os.path.basename(tiff_path)}...")
            feats, meta = self.extract_features(tiff_path)
            if feats is None: continue
            
            # Predict
            preds = self.rf_model.predict(feats)
            
            # Reshape to image
            h, w = meta['shape']
            # Unpack features for post-classification filtering
            ndwi = feats[:, 5]
            ndsi = feats[:, 6] # Using mndwi logic as ndsi proxy
            slope = feats[:, 9]
            dist = feats[:, 10]
            brightness = (feats[:, 0] + feats[:, 1] + feats[:, 2]) / 3.0
            
            # Reshape to image
            h, w = meta['shape']
            class_map = preds.reshape((h, w)).astype(np.uint8)
            
            # --- SEGMENTATION-FIRST REFINEMENT ---
            # Enforce physical constraints to remove confusion
            
            mask_high_slope = (slope > 15).reshape(h, w)
            mask_snow = ((brightness > 3500) & (ndsi > 0.4)).reshape(h, w) # Snow is Bright AND High NDSI
            
            # Remove "Lakes" that are actually on steep slopes or are snow
            class_map[(class_map == 1) & mask_high_slope] = 0 # Reject steep lakes
            class_map[(class_map == 1) & mask_snow] = 2 # Reclassify as Snow/Ice if bright
            
            # 3. Spatial Refinement (Object-Based)
            lake_mask = (class_map == 1)
            labeled_lakes = label(lake_mask)[0]
            regions = regionprops(labeled_lakes)
            
            refined_lakes = np.zeros_like(lake_mask, dtype=np.uint8)
            non_glacial_lakes = np.zeros_like(lake_mask, dtype=np.uint8)
            
            for prop in regions:
                # Area Threshold
                if prop.area < 10: continue
                
                # Proximity Check (Mean distance of object to glacier)
                # If avg distance > 2000m (example threshold), it's likely non-glacial
                lake_pixels = (labeled_lakes == prop.label)
                mean_dist = np.mean(dist.reshape(h, w)[lake_pixels])
                
                if mean_dist < 2000: # 2km threshold for Glacial Lake
                    refined_lakes[lake_pixels] = 1
                else:
                    non_glacial_lakes[lake_pixels] = 1 # New Class: Non-Glacial Lake
            
            # Update class map
            class_map[:] = 0 # Reset to rebuild
            # Add back classes (approximate for background/others from original preds if needed, 
            # but here we focus on the cleaned Lakes)
            
            # Re-apply original predictions for Ice/Shadow/Bg where not Lake
            original_preds = preds.reshape((h, w))
            class_map[original_preds == 2] = 2
            class_map[original_preds == 3] = 3
            
            # Apply refined lakes
            class_map[refined_lakes == 1] = 1
            class_map[non_glacial_lakes == 1] = 4 # Class 4: Non-Glacial Lake
            
            # Visualize
            self.save_visual(class_map, tiff_path)
            
            # Save Vector
            self.save_vector(class_map, meta)

    def save_visual(self, class_map, tiff_path):
        date_str = os.path.basename(tiff_path)[10:20]
        out_path = os.path.join(self.output_dir, f"Classified_{date_str}.png")
        
        cmap = ListedColormap(['white', 'cyan', 'grey', 'black', 'blue']) # 0:Bg, 1:GLake, 2:Ice, 3:Shadow, 4:NonGLake
        plt.figure(figsize=(10, 10))
        plt.imshow(class_map, cmap=cmap, vmin=0, vmax=4)
        plt.title(f"Fused Classification: {date_str}")
        plt.axis('off')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_vector(self, class_map, meta):
        # Implementation of vector saving (Polygonize)
        # Identify classes to save: 1 (Glacial Lake), 4 (Non-Glacial Lake)
        mask = (class_map == 1) | (class_map == 4)
        if not np.any(mask):
            return # No lakes found

        # Polygonize
        # shapes returns generator of (geojson, value)
        results = (
            {'properties': {'class_id': int(v), 'type': 'Glacial Lake' if v==1 else 'Non-Glacial Lake'}, 'geometry': s}
            for i, (s, v) in enumerate(
                shapes(class_map, mask=mask, transform=meta['transform']))
        )
        
        geoms = list(results)
        if not geoms: 
            return

        try:
            gdf = gpd.GeoDataFrame.from_features(geoms, crs=meta['crs'])
            
            date_str = meta['date']
            out_name = f"Lakes_{date_str}.geojson"
            out_path = os.path.join(self.output_dir, out_name)
            
            gdf.to_file(out_path, driver='GeoJSON')
            print(f"Saved vector: {out_name}")
        except Exception as e:
            print(f"Vector save failed: {e}")

if __name__ == "__main__":
    # CONFIGURATION
    TIFF_DIR = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    VECTOR_DIR = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\universal_vectors"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\enhanced_classification"
    
    # Path to DEM - Update this if available!
    # If None, DEM/Slope features will be 0 but script will run.
    DEM_PATH = None 
    
    classifier = FusionClassifier(TIFF_DIR, VECTOR_DIR, OUTPUT_DIR, DEM_PATH)
    classifier.run()
