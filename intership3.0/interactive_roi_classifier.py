"""
Interactive ROI Training Classifier
====================================
Click to mark training samples:
- LEFT CLICK = Lake
- RIGHT CLICK = Glacier
- MIDDLE CLICK (or Shift+Left) = Terrain/Shadow

Then the classifier learns from YOUR samples and classifies the whole image.

Instructions:
1. Image will open in a window
2. Click 5-10 pixels in each class (lake, glacier, terrain)
3. Close the window when done
4. Classifier will train and classify automatically

Author: Antigravity
Date: 2026-01-23
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from skimage import exposure
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')


class InteractiveROIClassifier:
    def __init__(self):
        self.training_samples = {
            'lake': [],      # Will store (b,g,r,n,s) tuples
            'glacier': [],
            'terrain': []
        }
        self.current_class = 'lake'
        self.bands = None
        self.profile = None
        self.fig = None
        self.ax = None
        
    def onclick(self, event):
        """Handle mouse clicks"""
        if event.xdata is None or event.ydata is None:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        if x < 0 or y < 0 or y >= self.bands['blue'].shape[0] or x >= self.bands['blue'].shape[1]:
            return
        
        # Get spectral signature at clicked point
        b = self.bands['blue'][y, x]
        g = self.bands['green'][y, x]
        r = self.bands['red'][y, x]
        n = self.bands['nir'][y, x]
        s = self.bands['swir'][y, x]
        
        # Determine class based on mouse button
        if event.button == 1:  # Left click = Lake
            class_name = 'lake'
            color = 'blue'
        elif event.button == 3:  # Right click = Glacier
            class_name = 'glacier'
            color = 'white'
        elif event.button == 2:  # Middle click = Terrain
            class_name = 'terrain'
            color = 'brown'
        else:
            return
        
        # Store sample
        self.training_samples[class_name].append((b, g, r, n, s))
        
        # Mark on image
        self.ax.plot(x, y, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)
        self.fig.canvas.draw()
        
        print(f"Added {class_name} sample at ({x}, {y}) - Total:{len(self.training_samples[class_name])}")
    
    def collect_training_data(self, tiff_path):
        """Load image and collect training samples interactively"""
        
        print(f"\n{'='*70}")
        print("INTERACTIVE ROI TRAINING - SAMPLE COLLECTION")
        print(f"{'='*70}")
        print("Loading image...")
        
        # Load bands
        with rasterio.open(tiff_path) as src:
            self.profile = src.profile
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
        
        # Scale to 0-1
        scale = 10000.0 if np.max(blue) > 2.0 else 1.0
        self.bands = {
            'blue': np.clip(blue / scale, 0, 1),
            'green': np.clip(green / scale, 0, 1),
            'red': np.clip(red / scale, 0, 1),
            'nir': np.clip(nir / scale, 0, 1),
            'swir': np.clip(swir / scale, 0, 1)
        }
        
        # Create RGB for display
        rgb = np.dstack([self.bands['red'], self.bands['green'], self.bands['blue']])
        # Enhance for better visibility
        rgb_display = exposure.equalize_adapthist(rgb, clip_limit=0.03)
        
        # Create interactive plot
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.ax.imshow(rgb_display)
        self.ax.set_title('Click to Mark Training Samples\nLEFT=Lake (Blue) | RIGHT=Glacier (White) | MIDDLE=Terrain (Brown)\n' +
                         'Mark 5-10 samples per class, then CLOSE this window',
                         fontsize=12, fontweight='bold', pad=15)
        self.ax.axis('off')
        
        # Connect mouse click event
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        print("\nINSTRUCTIONS:")
        print("  LEFT CLICK = Mark as LAKE (blue)")
        print("  RIGHT CLICK = Mark as GLACIER (white)")
        print("  MIDDLE CLICK = Mark as TERRAIN/SHADOW (brown)")
        print("\nMark 5-10 samples for each class, then close the window to continue...\n")
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n{'='*70}")
        print("Training samples collected:")
        print(f"  Lake: {len(self.training_samples['lake'])}")
        print(f"  Glacier: {len(self.training_samples['glacier'])}")
        print(f"  Terrain: {len(self.training_samples['terrain'])}")
        print(f"{'='*70}\n")
    
    def train_and_classify(self):
        """Train classifier on samples and classify whole image"""
        
        # Check if we have samples
        if len(self.training_samples['lake']) == 0:
            print("ERROR: No lake samples! Please mark some lake pixels.")
            return None
        if len(self.training_samples['glacier']) == 0:
            print("ERROR: No glacier samples! Please mark some glacier pixels.")
            return None
        if len(self.training_samples['terrain']) == 0:
            print("ERROR: No terrain samples! Please mark some terrain pixels.")
            return None
        
        print("Training classifier...")
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for sample in self.training_samples['terrain']:
            X_train.append(sample)
            y_train.append(0)  # Terrain = 0
        
        for sample in self.training_samples['glacier']:
            X_train.append(sample)
            y_train.append(1)  # Glacier = 1
        
        for sample in self.training_samples['lake']:
            X_train.append(sample)
            y_train.append(2)  # Lake = 2
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train Random Forest classifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        clf.fit(X_train, y_train)
        
        print("  Classifier trained successfully!")
        print("  Feature importances:")
        features = ['Blue', 'Green', 'Red', 'NIR', 'SWIR']
        for feat, imp in zip(features, clf.feature_importances_):
            print(f"    {feat}: {imp:.3f}")
        
        # Classify entire image
        print("\nClassifying entire image...")
        h, w = self.bands['blue'].shape
        
        # Stack all bands
        img_stack = np.dstack([
            self.bands['blue'],
            self.bands['green'],
            self.bands['red'],
            self.bands['nir'],
            self.bands['swir']
        ])
        
        # Reshape for prediction
        X = img_stack.reshape(-1, 5)
        
        # Predict
        y_pred = clf.predict(X)
        
        # Reshape back to image
        class_map = y_pred.reshape(h, w).astype(np.uint8)
        
        # Calculate statistics
        total = h * w
        terrain_pct = (np.sum(class_map == 0) / total) * 100
        glacier_pct = (np.sum(class_map == 1) / total) * 100
        lake_pct = (np.sum(class_map == 2) / total) * 100
        
        print(f"\nClassification Results:")
        print(f"  Terrain: {terrain_pct:.2f}%")
        print(f"  Glacier: {glacier_pct:.2f}%")
        print(f"  Lake: {lake_pct:.2f}%")
        
        return class_map
    
    def save_results(self, class_map, output_dir, date_str):
        """Save classification results"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create RGB for visualization
        rgb = np.dstack([self.bands['red'], self.bands['green'], self.bands['blue']])
        rgb_display = exposure.equalize_adapthist(rgb, clip_limit=0.03)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Original
        axes[0].imshow(rgb_display)
        axes[0].set_title('Enhanced RGB', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 2. Classification
        colors = ['#8B7355', '#FFFFFF', '#0080FF']
        cmap = ListedColormap(colors)
        axes[1].imshow(class_map, cmap=cmap, vmin=0, vmax=2)
        axes[1].set_title('ROI-Trained Classification', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        legend = [
            Patch(facecolor='#8B7355', label='Terrain/Shadow'),
            Patch(facecolor='#FFFFFF', label='Glacier'),
            Patch(facecolor='#0080FF', label='Lake')
        ]
        axes[1].legend(handles=legend, loc='lower right')
        
        # 3. Overlay
        overlay = rgb_display.copy()
        overlay[class_map == 1] = [0.9, 1.0, 1.0]  # Cyan for glacier
        overlay[class_map == 2] = [0, 0.6, 1.0]    # Blue for lake
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay Check', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'Interactive ROI Training Results - {date_str}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        viz_path = os.path.join(output_dir, f"ROI_Classification_{date_str}.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Saved visualization: {viz_path}")
        
        # Save GeoTIFF
        tif_path = os.path.join(output_dir, f"ROI_Classification_{date_str}.tif")
        prof_out = self.profile.copy()
        prof_out.update({'dtype': rasterio.uint8, 'count': 1, 'compress': 'lzw'})
        with rasterio.open(tif_path, 'w', **prof_out) as dst:
            dst.write(class_map, 1)
        print(f"✓ Saved GeoTIFF: {tif_path}")
        
        # Save vectors
        for class_val, class_name in [(1, 'Glacier'), (2, 'Lake')]:
            mask = (class_map == class_val).astype(np.uint8)
            if np.any(mask):
                results = [
                    {'properties': {'class': class_name, 'date': date_str}, 'geometry': s}
                    for s, v in shapes(mask, mask=mask.astype(bool), transform=self.profile['transform'])
                ]
                if results:
                    gdf = gpd.GeoDataFrame.from_features(results, crs=self.profile['crs'])
                    vec_path = os.path.join(output_dir, f"ROI_{class_name}_{date_str}.geojson")
                    gdf.to_file(vec_path, driver='GeoJSON')
                    print(f"✓ Saved vector: {vec_path}")


def run_interactive_classification(tiff_path, output_dir):
    """Run the interactive ROI training workflow"""
    
    date_str = os.path.basename(tiff_path)[10:20]
    if len(date_str) < 5:
        date_str = "test"
    
    print(f"\n{'='*70}")
    print("INTERACTIVE ROI TRAINING CLASSIFIER")
    print(f"{'='*70}")
    print(f"Image: {os.path.basename(tiff_path)}")
    print(f"Date: {date_str}")
    print(f"{'='*70}\n")
    
    # Create classifier
    classifier = InteractiveROIClassifier()
    
    # Step 1: Collect training samples
    classifier.collect_training_data(tiff_path)
    
    # Step 2: Train and classify
    class_map = classifier.train_and_classify()
    
    if class_map is not None:
        # Step 3: Save results
        classifier.save_results(class_map, output_dir, date_str)
        
        print(f"\n{'='*70}")
        print("COMPLETE!")
        print(f"Output: {output_dir}")
        print(f"{'='*70}\n")
    else:
        print("\nClassification failed. Please ensure you marked samples for all 3 classes.")


if __name__ == "__main__":
    TIFF_DIR = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\roi_training"
    
    import glob
    tiff_files = sorted(glob.glob(os.path.join(TIFF_DIR, "*.tif")))
    
    if tiff_files:
        run_interactive_classification(tiff_files[0], OUTPUT_DIR)
    else:
        print("No TIFF files found!")
