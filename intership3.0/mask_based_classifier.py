"""
Mask-Based Glacier Classifier with Correction Tool
===================================================
Workflow:
1. Load your hand-marked lake mask (blue pixels = lakes)
2. Automatically extract training samples
3. Train classifier and classify
4. Show results - you can draw rectangles around errors if needed

Author: Antigravity
Date: 2026-01-23
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.patches import Rectangle as MPLRectangle
from sklearn.ensemble import RandomForestClassifier
from skimage import exposure
from skimage.io import imread
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')


class MaskBasedClassifier:
    def __init__(self):
        self.clf = None
        self.bands = None
        self.profile = None
        
    def load_bands(self, tiff_path):
        """Load Sentinel-2 bands"""
        print("Loading Sentinel-2 bands...")
        
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
        print(f"  Scaling by {scale}")
        
        self.bands = {
            'blue': np.clip(blue / scale, 0, 1),
            'green': np.clip(green / scale, 0, 1),
            'red': np.clip(red / scale, 0, 1),
            'nir': np.clip(nir / scale, 0, 1),
            'swir': np.clip(swir / scale, 0, 1)
        }
        
        return self.bands
    
    def extract_samples_from_mask(self, mask_path, num_samples_per_class=500):
        """
        Extract training samples from your hand-marked mask
        Assumes: Blue pixels (RGB where B > R and B > G) = Lakes
        """
        print(f"\nExtracting training samples from: {os.path.basename(mask_path)}")
        
        # Load mask image
        mask_img = imread(mask_path)
        
        # Resize mask to match Sentinel-2 if needed
        h, w = self.bands['blue'].shape
        if mask_img.shape[0] != h or mask_img.shape[1] != w:
            from skimage.transform import resize
            print(f"  Resizing mask from {mask_img.shape[:2]} to {(h, w)}")
            mask_img = resize(mask_img, (h, w), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        
        # Extract RGB channels
        if len(mask_img.shape) == 3:
            r_mask = mask_img[:, :, 0]
            g_mask = mask_img[:, :, 1]
            b_mask = mask_img[:, :, 2]
        else:
            # Grayscale - assume marked area is bright
            r_mask = mask_img
            g_mask = mask_img
            b_mask = mask_img
        
        # Identify lake pixels (marked in blue)
        # Blue dominant: B > R and B > G
        lake_mask = (b_mask > r_mask + 20) & (b_mask > g_mask + 20) & (b_mask > 100)
        
        # Identify glacier pixels (marked in white or cyan)
        # White: R ≈ G ≈ B and all > 200
        # Cyan: B > 200, G > 200, R < 150
        glacier_mask = (
            ((r_mask > 200) & (g_mask > 200) & (b_mask > 200)) |  # White
            ((b_mask > 200) & (g_mask > 200) & (r_mask < 150))    # Cyan
        )
        
        # Everything else is terrain/shadow
        terrain_mask = ~(lake_mask | glacier_mask)
        
        print(f"  Found marked pixels:")
        print(f"    Lake: {np.sum(lake_mask):,} pixels")
        print(f"    Glacier: {np.sum(glacier_mask):,} pixels")
        print(f"    Terrain: {np.sum(terrain_mask):,} pixels")
        
        # Sample randomly from each class
        X_train = []
        y_train = []
        
        # Sample terrain
        terrain_coords = np.argwhere(terrain_mask)
        if len(terrain_coords) > num_samples_per_class:
            terrain_coords = terrain_coords[np.random.choice(len(terrain_coords), num_samples_per_class, replace=False)]
        for y, x in terrain_coords:
            sample = [self.bands['blue'][y,x], self.bands['green'][y,x], self.bands['red'][y,x],
                     self.bands['nir'][y,x], self.bands['swir'][y,x]]
            X_train.append(sample)
            y_train.append(0)
        
        # Sample glacier
        glacier_coords = np.argwhere(glacier_mask)
        if len(glacier_coords) > num_samples_per_class:
            glacier_coords = glacier_coords[np.random.choice(len(glacier_coords), num_samples_per_class, replace=False)]
        for y, x in glacier_coords:
            sample = [self.bands['blue'][y,x], self.bands['green'][y,x], self.bands['red'][y,x],
                     self.bands['nir'][y,x], self.bands['swir'][y,x]]
            X_train.append(sample)
            y_train.append(1)
        
        # Sample lake
        lake_coords = np.argwhere(lake_mask)
        if len(lake_coords) > num_samples_per_class:
            lake_coords = lake_coords[np.random.choice(len(lake_coords), num_samples_per_class, replace=False)]
        for y, x in lake_coords:
            sample = [self.bands['blue'][y,x], self.bands['green'][y,x], self.bands['red'][y,x],
                     self.bands['nir'][y,x], self.bands['swir'][y,x]]
            X_train.append(sample)
            y_train.append(2)
        
        print(f"\n  Training samples extracted:")
        print(f"    Terrain: {np.sum(np.array(y_train) == 0)}")
        print(f"    Glacier: {np.sum(np.array(y_train) == 1)}")
        print(f"    Lake: {np.sum(np.array(y_train) == 2)}")
        
        return np.array(X_train), np.array(y_train)
    
    def train_classifier(self, X_train, y_train):
        """Train Random Forest classifier"""
        print("\nTraining Random Forest classifier...")
        
        self.clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        self.clf.fit(X_train, y_train)
        
        print("  Training complete!")
        print("  Feature importances:")
        features = ['Blue', 'Green', 'Red', 'NIR', 'SWIR']
        for feat, imp in zip(features, self.clf.feature_importances_):
            print(f"    {feat}: {imp:.3f}")
    
    def classify_image(self):
        """Classify entire image"""
        print("\nClassifying entire image...")
        
        h, w = self.bands['blue'].shape
        
        # Stack bands
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
        y_pred = self.clf.predict(X)
        
        # Reshape back
        class_map = y_pred.reshape(h, w).astype(np.uint8)
        
        # Stats
        total = h * w
        terrain_pct = (np.sum(class_map == 0) / total) * 100
        glacier_pct = (np.sum(class_map == 1) / total) * 100
        lake_pct = (np.sum(class_map == 2) / total) * 100
        
        print(f"\n  Classification Results:")
        print(f"    Terrain: {terrain_pct:.2f}%")
        print(f"    Glacier: {glacier_pct:.2f}%")
        print(f"    Lake: {lake_pct:.2f}%")
        
        return class_map
    
    def save_results(self, class_map, output_dir, date_str):
        """Save results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create RGB
        rgb = np.dstack([self.bands['red'], self.bands['green'], self.bands['blue']])
        rgb_display = exposure.equalize_adapthist(rgb, clip_limit=0.03)
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        # 1. Original
        axes[0].imshow(rgb_display)
        axes[0].set_title('Enhanced RGB', fontsize=13, fontweight='bold')
        axes[0].axis('off')
        
        # 2. Classification
        colors = ['#654321', '#F0F0F0', '#0080FF']
        cmap = ListedColormap(colors)
        axes[1].imshow(class_map, cmap=cmap, vmin=0, vmax=2)
        axes[1].set_title('ML Classification (From Your Mask)', fontsize=13, fontweight='bold')
        axes[1].axis('off')
        
        legend = [
            Patch(facecolor='#654321', label='Terrain/Shadow'),
            Patch(facecolor='#F0F0F0', label='Glacier/Snow'),
            Patch(facecolor='#0080FF', label='Lake')
        ]
        axes[1].legend(handles=legend, loc='lower right', fontsize=10)
        
        # 3. Overlay
        overlay = rgb_display.copy()
        overlay[class_map == 1] = [0.9, 1.0, 1.0]
        overlay[class_map == 2] = [0, 0.6, 1.0]
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay Check', fontsize=13, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'Mask-Based Classification - {date_str}', fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        viz_path = os.path.join(output_dir, f"Mask_Classification_{date_str}.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Saved: {viz_path}")
        
        # GeoTIFF
        tif_path = os.path.join(output_dir, f"Classification_{date_str}.tif")
        prof_out = self.profile.copy()
        prof_out.update({'dtype': rasterio.uint8, 'count': 1, 'compress': 'lzw'})
        with rasterio.open(tif_path, 'w', **prof_out) as dst:
            dst.write(class_map, 1)
        print(f"✓ Saved: {tif_path}")
        
        # Vectors
        for class_val, class_name in [(1, 'Glacier'), (2, 'Lake')]:
            mask = (class_map == class_val).astype(np.uint8)
            if np.any(mask):
                results = [
                    {'properties': {'class': class_name, 'date': date_str}, 'geometry': s}
                    for s, v in shapes(mask, mask=mask.astype(bool), transform=self.profile['transform'])
                ]
                if results:
                    gdf = gpd.GeoDataFrame.from_features(results, crs=self.profile['crs'])
                    vec_path = os.path.join(output_dir, f"{class_name}_{date_str}.geojson")
                    gdf.to_file(vec_path, driver='GeoJSON')
                    print(f"✓ Saved: {vec_path}")


def run_mask_based_classification(tiff_path, mask_path, output_dir):
    """Run classification from hand-marked mask"""
    
    date_str = os.path.basename(tiff_path)[10:20]
    if len(date_str) < 5:
        date_str = "test"
    
    print(f"\n{'='*70}")
    print("MASK-BASED GLACIER CLASSIFIER")
    print(f"{'='*70}")
    print(f"Sentinel-2 Image: {os.path.basename(tiff_path)}")
    print(f"Lake Mask: {os.path.basename(mask_path)}")
    print(f"Date: {date_str}")
    print(f"{'='*70}")
    
    # Create classifier
    classifier = MaskBasedClassifier()
    
    # Load bands
    classifier.load_bands(tiff_path)
    
    # Extract training samples from your mask
    X_train, y_train = classifier.extract_samples_from_mask(mask_path)
    
    if len(X_train) == 0 or len(set(y_train)) < 3:
        print("\nERROR: Could not extract enough training samples!")
        print("Make sure your mask image has:")
        print("  - Blue pixels for lakes")
        print("  - White/Cyan pixels for glaciers")
        print("  - Other colors for terrain")
        return
    
    # Train
    classifier.train_classifier(X_train, y_train)
    
    # Classify
    class_map = classifier.classify_image()
    
    # Save
    classifier.save_results(class_map, output_dir, date_str)
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # CONFIGURATION
    TIFF_PATH = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\Sentinel2_2021-11-08.tif"  # Adjust to your file
    MASK_PATH = r"f:\NIDM-SRM Drone Project Docs\intership3.0\lake_mask.png"  # YOUR hand-marked image
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\mask_based_classification"
    
    # Check if mask exists
    if not os.path.exists(MASK_PATH):
        print(f"\nERROR: Mask file not found!")
        print(f"Expected: {MASK_PATH}")
        print("\nPlease provide the path to your blue-marked lake image.")
        print("You can edit the MASK_PATH variable in this script.\n")
    else:
        run_mask_based_classification(TIFF_PATH, MASK_PATH, OUTPUT_DIR)
