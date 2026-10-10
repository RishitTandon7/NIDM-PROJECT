"""
Precise 3-Class Glacier Classifier (v18) - Drag-to-Paint UX
===========================================================
Improvements over v17:
- DRAG-TO-PAINT (UX Fix):
  - Replaced "Click-to-Paint" with "Drag-to-Paint".
  - You can now HOLD the mouse button and sweep over areas to correct them.
  - Much faster for marking large lakes or shadow regions.
- LARGER BRUSH:
  - Default radius increased to 10px (was 5px) for better coverage.
- BATCH PROCESSING:
  - Workflow remains: Fix 1 image -> Auto-process dataset.

Class 0: Non-Glacier (Terrain/Shadow)
Class 1: Glacier (Ice/Snow)
Class 2: Glacial Lake

Author: Antigravity
Date: 2026-01-23
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.features import shapes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from skimage import exposure, morphology
from scipy.ndimage import binary_opening
import geopandas as gpd
import joblib
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

# Global list to store user clicks [row, col, class_id]
USER_POINTS = []

class PreciseGlacierClassifier:
    def __init__(self):
        self.model_path = "glacier_memory_model.pkl"
        self.rf_model = None
        # Try to load existing memory
        if os.path.exists(self.model_path):
            print(f"  [Memory] Found existing knowledge file: {self.model_path}")
    
    def scale_bands_globally(self, blue, green, red, nir, swir):
        stack = np.dstack([blue, green, red, nir])
        if np.max(stack) > 2.0:
            scale_factor = 10000.0
        else:
            scale_factor = 1.0
            
        return (
            np.clip(blue / scale_factor, 0, 1),
            np.clip(green / scale_factor, 0, 1),
            np.clip(red / scale_factor, 0, 1),
            np.clip(nir / scale_factor, 0, 1),
            np.clip(swir / scale_factor, 0, 1)
        )

    def enhance_for_viz(self, band):
        return exposure.equalize_adapthist(band, clip_limit=0.03)
    
    def load_bands(self, tiff_path):
        with rasterio.open(tiff_path) as src:
            profile = src.profile
            blue = src.read(1).astype(np.float32)
            green = src.read(2).astype(np.float32)
            red = src.read(3).astype(np.float32)
            nir = src.read(4).astype(np.float32)
            if src.count >= 9:
                swir = src.read(9).astype(np.float32) 
            elif src.count >= 5:
                swir = src.read(5).astype(np.float32)
            else:
                swir = np.zeros_like(blue)
        return blue, green, red, nir, swir, profile

    def get_features(self, b, g, r, n, s):
        """Extract features for Machine Learning"""
        eps = 1e-10
        ndwi = (g - n) / (g + n + eps)
        mndwi = (g - s) / (g + s + eps)
        ndvi = (n - r) / (n + r + eps)
        brightness = (b + g + r) / 3.0
        saturation = (np.maximum(b, np.maximum(g, r)) - np.minimum(b, np.minimum(g, r))) / (np.maximum(b, np.maximum(g, r)) + eps)
        
        # Stack features for ML: [Rows*Cols, n_features]
        features_stack = np.dstack([b, g, r, n, s, ndwi, mndwi, ndvi, brightness, saturation])
        h, w, c = features_stack.shape
        return features_stack.reshape((-1, c)), (h, w)

    def rule_based_classify(self, blue, green, red, nir, swir):
        """The v15 Physics Logic (Used as a baseline/teacher)"""
        b, g, r, n, s = self.scale_bands_globally(blue, green, red, nir, swir)
        eps = 1e-10
        
        ndwi = (g - n) / (g + n + eps)
        mndwi = (g - s) / (g + s + eps)
        ndsi = (g - s) / (g + s + eps)
        ndvi = (n - r) / (n + r + eps)
        brightness = (b + g + r) / 3.0
        
        h, w = b.shape
        class_map = np.zeros((h, w), dtype=np.uint8)
        
        # 1. Glacier
        glacier_mask = (ndsi > 0.4) & (n > 0.11) & (brightness > 0.35) & (ndvi < 0.1)
        
        # 2. Water (Precision v15 rules)
        index_check = (mndwi > 0.1) & (ndwi > -0.05)
        spectral_shape_check = (g > r * 1.05) | (b > r * 1.05)
        brightness_floor = (b > 0.025)
        absorption_check = (n < 0.3) & (s < 0.1)
        
        water_mask = index_check & spectral_shape_check & brightness_floor & absorption_check & (~glacier_mask)
        
        # Cleanup
        glacier_mask = morphology.remove_small_objects(glacier_mask, min_size=20)
        water_mask = morphology.remove_small_objects(water_mask, min_size=20)
        
        class_map[glacier_mask] = 1
        class_map[water_mask] = 2
        
        return class_map

    def interactive_correction(self, class_map, rgb, date_str):
        """Opens a window for the user to mark mistakes using a Drag-to-Paint Brush"""
        print("\n" + "="*60)
        print("INTERACTIVE CORRECTION MODE (Paint Brush)")
        print("="*60)
        print("1. HOLD LEFT CLICK   -> Paint WATER")
        print("2. HOLD RIGHT CLICK  -> Paint BACKGROUND (Shadow/Land)")
        print("3. HOLD MIDDLE CLICK -> Paint GLACIER")
        print("4. Close window -> SAVE MODEL & Apply to Batch")
        print("="*60)

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_title(f"Teaching Mode: {date_str}\nDRAG mouse to Paint (Radius=10px)", fontsize=14)
        
        # Show overlay
        overlay = rgb.copy()
        mask_g = class_map == 1
        mask_w = class_map == 2
        
        # Tint existing classes
        overlay[mask_g, 0] = np.clip(overlay[mask_g, 0] + 0.4, 0, 1) # Red tint for Glacier
        overlay[mask_w, 2] = np.clip(overlay[mask_w, 2] + 0.4, 0, 1) # Blue tint for Water
        
        ax.imshow(overlay)
        
        # Storage for display points
        points_w, = ax.plot([], [], 'bo', markersize=2, alpha=0.5, label='Added Water') 
        points_b, = ax.plot([], [], 'rx', markersize=2, alpha=0.5, label='Added Background')
        points_g, = ax.plot([], [], 'c^', markersize=2, alpha=0.5, label='Added Glacier')
        ax.legend()
        
        # Context to track dragging state
        ctx = {
            'is_drawing': False,
            'current_class': None,
            'brush_radius': 10 # Larger brush for smoother UX
        }

        def paint(x, y, cls):
            # BRUSH LOGIC: Add points in a circle
            r_center, c_center = y, x
            radius = ctx['brush_radius']
            
            # Simple square box for speed in loops, or circle mask
            y_grid, x_grid = np.ogrid[-radius: radius+1, -radius: radius+1]
            mask = x_grid**2 + y_grid**2 <= radius**2
            
            # Add new points
            new_pts = []
            for r_offset, c_offset in zip(*np.where(mask)):
                r = r_center + r_offset - radius
                c = c_center + c_offset - radius
                
                # Boundary check
                if 0 <= r < class_map.shape[0] and 0 <= c < class_map.shape[1]:
                    USER_POINTS.append([r, c, cls])
                    new_pts.append([r, c])
            
            # Optional: Real-time update (can be slow if painting fast, but useful feedback)
            if len(new_pts) > 0:
                update_viz()

        def update_viz():
            # Only update every N points to save FPS if needed, but for now simple update
            # Filtering large list is slow, so we might skip frequent updates in real-time
            # For better UX, we just update the specific handle
            pass 
            # Note: Matplotlib real-time drag plotting can be laggy with thousands of points.
            # We will rely on the final refresh or interval refresh.
            # To verify painting, we'll just plot the last batch.

        def on_press(event):
            if event.xdata is None or event.ydata is None: return
            ctx['is_drawing'] = True
            
            if event.button == 1: ctx['current_class'] = 2   # Left
            elif event.button == 3: ctx['current_class'] = 0 # Right
            elif event.button == 2: ctx['current_class'] = 1 # Middle
            else: 
                ctx['is_drawing'] = False
                return

            print(f"  Started Painting Class {ctx['current_class']}")
            paint(int(event.xdata), int(event.ydata), ctx['current_class'])

        def on_move(event):
            if not ctx['is_drawing'] or event.xdata is None or event.ydata is None: return
            paint(int(event.xdata), int(event.ydata), ctx['current_class'])

        def on_release(event):
            ctx['is_drawing'] = False
            # Update visualization on release to avoid lag while dragging
            w_pts = np.array([p for p in USER_POINTS if p[2] == 2])
            b_pts = np.array([p for p in USER_POINTS if p[2] == 0])
            g_pts = np.array([p for p in USER_POINTS if p[2] == 1])
            
            if len(w_pts) > 0: points_w.set_data(w_pts[:, 1], w_pts[:, 0])
            if len(b_pts) > 0: points_b.set_data(b_pts[:, 1], b_pts[:, 0])
            if len(g_pts) > 0: points_g.set_data(g_pts[:, 1], g_pts[:, 0])
            fig.canvas.draw()

        # Connect events
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_move)
        fig.canvas.mpl_connect('button_release_event', on_release)
        
        plt.show()
        return USER_POINTS

    def train_and_predict(self, X_flat, shape, initial_map):
        """Trains Random Forest on user corrections + confident physics pixels"""
        h, w = shape
        y_flat = initial_map.ravel()
        
        train_features = []
        train_labels = []
        
        # A. Add User Points (High Weight)
        if USER_POINTS:
            print(f"\n  [Learning] Incorporating {len(USER_POINTS)} manual correction points...")
            # Reduced weight since brush adds many points
            weight = 5 
            for r, c, label in USER_POINTS:
                idx = r * w + c
                if 0 <= idx < len(X_flat):
                    feat = X_flat[idx]
                    for _ in range(weight):
                        train_features.append(feat)
                        train_labels.append(label)
        
        # B. Add "Safe" Auto-Points
        print("  [Learning] Sampling baseline structure...")
        num_samples = 5000 
        indices = np.random.choice(len(X_flat), num_samples, replace=False)
        for idx in indices:
            train_features.append(X_flat[idx])
            train_labels.append(y_flat[idx])
            
        train_features = np.array(train_features)
        train_labels = np.array(train_labels)
        
        # 2. Train Model
        print("  [Learning] Training Random Forest Memory...")
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        rf.fit(train_features, train_labels)
        
        # 3. Predict
        print("  [Learning] Applying new logic...")
        prediction = rf.predict(X_flat)
        new_map = prediction.reshape(shape).astype(np.uint8)
        
        # 4. Save Memory
        self.rf_model = rf
        joblib.dump(rf, self.model_path)
        print(f"  [Memory] Logic saved to {self.model_path}")
        
        return new_map

    def save_vectors(self, class_map, profile, output_dir, date_str):
        transform = profile['transform']
        for class_val, class_name in [(1, 'Glacier'), (2, 'Lake')]:
            mask = (class_map == class_val).astype(np.uint8)
            if np.any(mask):
                results = ({'properties': {'class': class_name, 'date': date_str}, 'geometry': s}
                    for s, v in shapes(mask, mask=mask.astype(bool), transform=transform))
                geoms = list(results)
                if geoms:
                    gdf = gpd.GeoDataFrame.from_features(geoms, crs=profile['crs'])
                    out_path = os.path.join(output_dir, f"{class_name}_{date_str}.geojson")
                    gdf.to_file(out_path, driver='GeoJSON')
                    print(f"✓ Saved {class_name} vector")
    
    def create_visualization(self, class_map, rgb_norm, date_str, output_path):
        fig, ax = plt.subplots(figsize=(15, 10))
        colors = ['#444444', '#FFFFFF', '#0099FF'] 
        cmap = ListedColormap(colors)
        ax.imshow(class_map, cmap=cmap, vmin=0, vmax=2)
        ax.set_title(f'Learned Classification - {date_str}', fontweight='bold')
        ax.axis('off')
        
        legend_elements = [
            Patch(facecolor='#444444', label='Background'),
            Patch(facecolor='#FFFFFF', label='Glacier'),
            Patch(facecolor='#0099FF', label='Glacial Lake')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Map: {output_path}")

def batch_process(tiff_dir, output_dir):
    import glob
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "*.tif")))
    
    if not tiff_files:
        print("No TIFF files found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found {len(tiff_files)} images. Starting Batch Process...")
    
    classifier = PreciseGlacierClassifier()
    
    # --- PHASE 1: TRAIN ON FIRST IMAGE ---
    first_img = tiff_files[0]
    print(f"\n[{os.path.basename(first_img)}] === TRAINING PHASE ===")
    
    # Load and Prepare
    b, g, r, n, s, prof = classifier.load_bands(first_img)
    b_n, g_n, r_n, n_n, s_n = classifier.scale_bands_globally(b, g, r, n, s)
    X_flat, shape = classifier.get_features(b_n, g_n, r_n, n_n, s_n)
    
    # Initial Guess
    print("  Generating initial guess using physics...")
    initial_map = classifier.rule_based_classify(b, g, r, n, s)
    
    # Interactive Correction
    rgb_viz = np.dstack([classifier.enhance_for_viz(r_n), 
                         classifier.enhance_for_viz(g_n), 
                         classifier.enhance_for_viz(b_n)])
    
    user_clicks = classifier.interactive_correction(initial_map, rgb_viz, "Training_Image")
    
    # Finalize Model
    if user_clicks:
        final_map = classifier.train_and_predict(X_flat, shape, initial_map)
    else:
        print("  No corrections made. Using physics model as baseline.")
        final_map = initial_map
        # Train model anyway so we have a .pkl for the batch
        classifier.train_and_predict(X_flat, shape, initial_map)

    # Save First Image
    date_str = os.path.basename(first_img)[10:20]
    classifier.create_visualization(final_map, rgb_viz, date_str, os.path.join(output_dir, f"Result_{date_str}.png"))
    classifier.save_vectors(final_map, prof, output_dir, date_str)
    
    # --- PHASE 2: AUTOMATIC BATCH ---
    if len(tiff_files) > 1:
        print(f"\n=== BATCH PROCESSING ({len(tiff_files)-1} images) ===")
        print("Applying learned model to remaining dataset...")
        
        # Reload model just to be safe
        classifier.rf_model = joblib.load(classifier.model_path)
        
        for tiff_path in tiff_files[1:]:
            fname = os.path.basename(tiff_path)
            print(f"\nProcessing: {fname}")
            date_str = fname[10:20]
            
            # Load
            b, g, r, n, s, prof = classifier.load_bands(tiff_path)
            b_n, g_n, r_n, n_n, s_n = classifier.scale_bands_globally(b, g, r, n, s)
            X_flat, shape = classifier.get_features(b_n, g_n, r_n, n_n, s_n)
            
            # Predict (No interaction)
            prediction = classifier.rf_model.predict(X_flat)
            class_map = prediction.reshape(shape).astype(np.uint8)
            
            # Save
            rgb_viz = np.dstack([classifier.enhance_for_viz(r_n), classifier.enhance_for_viz(g_n), classifier.enhance_for_viz(b_n)])
            classifier.create_visualization(class_map, rgb_viz, date_str, os.path.join(output_dir, f"Result_{date_str}.png"))
            classifier.save_vectors(class_map, prof, output_dir, date_str)
            
    print("\nBatch Processing Complete.")

if __name__ == "__main__":
    TIFF_DIR = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\precise_test_interactive"
    
    batch_process(TIFF_DIR, OUTPUT_DIR)