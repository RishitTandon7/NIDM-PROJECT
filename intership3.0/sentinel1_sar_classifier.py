"""
Sentinel-1 SAR Glacier Classifier
==================================
Classifies Sentinel-1 SAR data into 3 classes:
  - Class 0: Non-Glacier (Terrain/Rock/Shadow)
  - Class 1: Glacier (Ice/Snow)
  - Class 2: Glacial Lake (Water bodies)

SAR-Specific Features Used:
  - VV and VH backscatter intensities
  - VV/VH ratio (polarimetric measure)
  - Cross-polarization difference
  - Entropy (texture/roughness)
  - Local incidence angle normalization

Scientific Basis:
  - Water bodies: Very low backscatter (specular reflection), VV/VH ratio typically lower
  - Glaciers/Ice: Medium backscatter with characteristic polarimetric response
  - Dry terrain: High backscatter with rough surface scattering

Author: Antigravity
Date: 2026-01-30
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.features import shapes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from skimage import exposure, morphology, filters
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import uniform_filter, gaussian_filter
import geopandas as gpd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Global storage for user corrections
USER_POINTS = []


class Sentinel1GlacierClassifier:
    """
    Main classifier for Sentinel-1 SAR data using SAR-specific features
    and interactive machine learning correction.
    """
    
    def __init__(self):
        self.model_path = "sentinel1_glacier_model.pkl"
        self.scaler_path = "sentinel1_scaler.pkl"
        self.rf_model = None
        self.scaler = None
        
        # Try to load existing model
        if os.path.exists(self.model_path):
            print(f"  [Memory] Found existing model: {self.model_path}")
            self.rf_model = joblib.load(self.model_path)
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
    
    def load_sar_bands(self, tiff_path):
        """Load Sentinel-1 VV, VH, and angle bands"""
        with rasterio.open(tiff_path) as src:
            profile = src.profile
            vv = src.read(1).astype(np.float32)  # VV polarization
            vh = src.read(2).astype(np.float32)  # VH polarization
            angle = src.read(3).astype(np.float32)  # Incidence angle
        
        return vv, vh, angle, profile
    
    def convert_to_db(self, linear_data):
        """Convert linear SAR values to dB scale"""
        eps = 1e-10
        db = 10 * np.log10(np.maximum(linear_data, eps))
        return db
    
    def normalize_backscatter(self, vv, vh, angle):
        """
        Apply angle normalization to correct for incidence angle effects.
        Reference angle is typically 35-40 degrees for Sentinel-1.
        """
        ref_angle = 35.0  # Reference incidence angle in degrees
        angle_rad = np.radians(angle)
        ref_angle_rad = np.radians(ref_angle)
        
        # Simple cosine normalization
        cos_ratio = np.cos(ref_angle_rad) / np.cos(angle_rad + 1e-10)
        
        vv_norm = vv * cos_ratio
        vh_norm = vh * cos_ratio
        
        return vv_norm, vh_norm
    
    def compute_texture_features(self, band, window_size=7):
        """
        Compute texture features using local statistics and GLCM-based measures.
        These help distinguish rough surfaces (glaciers, terrain) from smooth ones (lakes).
        """
        # Local mean and standard deviation
        local_mean = uniform_filter(band, size=window_size)
        local_sqmean = uniform_filter(band ** 2, size=window_size)
        local_var = np.maximum(local_sqmean - local_mean ** 2, 0)
        local_std = np.sqrt(local_var)
        
        # Coefficient of variation (speckle measure)
        coeff_var = local_std / (np.abs(local_mean) + 1e-10)
        
        # Edge strength using Sobel filter
        edge = filters.sobel(band)
        
        return local_mean, local_std, coeff_var, edge
    
    def compute_sar_features(self, vv, vh, angle):
        """
        Extract SAR-specific features for classification.
        
        Feature Set:
        1. VV backscatter (dB)
        2. VH backscatter (dB)
        3. VV/VH ratio (polarimetric index)
        4. Cross-polarization difference (VV - VH in dB)
        5. Radar Vegetation Index (approximation)
        6. Local texture features (mean, std, CoV)
        7. Edge strength
        """
        # Normalize for angle effects
        vv_norm, vh_norm = self.normalize_backscatter(vv, vh, angle)
        
        # Convert to dB
        vv_db = self.convert_to_db(vv_norm)
        vh_db = self.convert_to_db(vh_norm)
        
        # Cap extreme values
        vv_db = np.clip(vv_db, -30, 10)
        vh_db = np.clip(vh_db, -35, 5)
        
        # Polarimetric features
        eps = 1e-10
        vv_vh_ratio = vv_db - vh_db  # Ratio in dB
        cross_pol_diff = vv_norm / (vh_norm + eps)  # Linear ratio
        
        # Radar Vegetation Index (RVI) - helps distinguish ice from vegetation
        rvi = (4 * vh_norm) / (vv_norm + vh_norm + eps)
        rvi = np.clip(rvi, 0, 2)
        
        # Texture features on VV (dominant for surface analysis)
        local_mean_vv, local_std_vv, coeff_var_vv, edge_vv = self.compute_texture_features(vv_db)
        
        # Texture features on VH
        local_mean_vh, local_std_vh, coeff_var_vh, edge_vh = self.compute_texture_features(vh_db)
        
        # Entropy-like measure (spatial heterogeneity)
        entropy = coeff_var_vv + coeff_var_vh
        
        # Stack all features
        features = np.dstack([
            vv_db,              # 0: VV backscatter (dB)
            vh_db,              # 1: VH backscatter (dB)
            vv_vh_ratio,        # 2: VV/VH ratio (dB)
            cross_pol_diff,     # 3: Cross-pol ratio (linear)
            rvi,                # 4: Radar Vegetation Index
            local_mean_vv,      # 5: Local mean VV
            local_std_vv,       # 6: Local std VV
            coeff_var_vv,       # 7: Coefficient of variation VV
            local_mean_vh,      # 8: Local mean VH
            local_std_vh,       # 9: Local std VH
            entropy,            # 10: Combined texture entropy
            edge_vv,            # 11: Edge strength VV
            angle               # 12: Incidence angle
        ])
        
        h, w, c = features.shape
        features_flat = features.reshape((-1, c))
        
        # Handle NaN/Inf
        features_flat = np.nan_to_num(features_flat, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_flat, (h, w), vv_db, vh_db
    
    def rule_based_classify(self, vv_db, vh_db, vv_vh_ratio, local_var):
        """
        Initial physics-based classification using SAR scattering properties.
        
        Water:   Very low backscatter (specular reflection) -> VV < -15 dB, VH < -20 dB
        Glacier: Medium backscatter with moderate polarization
        Terrain: Higher backscatter with rough surface signature
        """
        h, w = vv_db.shape
        class_map = np.zeros((h, w), dtype=np.uint8)
        
        # Compute additional features
        mean_backscatter = (vv_db + vh_db) / 2
        
        # ----- WATER (Glacial Lakes) Detection -----
        # Water has very low backscatter due to specular reflection
        # VV typically < -15 dB, VH < -22 dB for calm water
        water_vv = vv_db < -12
        water_vh = vh_db < -18
        water_low_texture = local_var < 2.0  # Very smooth
        
        water_mask = water_vv & water_vh & water_low_texture
        
        # ----- GLACIER (Ice/Snow) Detection -----
        # Glaciers: moderate backscatter, higher than water but can vary
        # Clean ice: lower backscatter (-15 to -8 dB)
        # Rough ice/firn: higher backscatter (-10 to 0 dB)
        glacier_backscatter = (vv_db > -15) & (vv_db < 2)
        glacier_moderate_vv_vh = (vv_vh_ratio > 3) & (vv_vh_ratio < 12)
        
        glacier_mask = glacier_backscatter & glacier_moderate_vv_vh & (~water_mask)
        
        # Assign classes
        class_map[glacier_mask] = 1  # Glacier
        class_map[water_mask] = 2    # Water/Lake
        
        # Clean up small regions
        class_map = morphology.remove_small_objects(
            class_map.astype(bool), min_size=50
        ).astype(np.uint8) * class_map
        
        return class_map
    
    def interactive_correction(self, class_map, vv_db, vh_db, date_str):
        """
        Opens an interactive window for user to paint corrections.
        Drag-to-paint interface for efficient marking.
        """
        global USER_POINTS
        USER_POINTS = []  # Reset for this session
        
        print("\n" + "=" * 60)
        print("INTERACTIVE SAR CORRECTION MODE (Paint Brush)")
        print("=" * 60)
        print("1. HOLD LEFT CLICK   -> Paint WATER (Lake)")
        print("2. HOLD RIGHT CLICK  -> Paint NON-GLACIER (Terrain/Shadow)")
        print("3. HOLD MIDDLE CLICK -> Paint GLACIER (Ice/Snow)")
        print("4. Close window      -> SAVE & Apply to Batch")
        print("=" * 60)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        fig.suptitle(f"SAR Classification Correction - {date_str}\nDrag to Paint (Radius=12px)", fontsize=14)
        
        # Panel 1: VV backscatter
        ax1 = axes[0]
        im1 = ax1.imshow(vv_db, cmap='gray', vmin=-25, vmax=0)
        ax1.set_title('VV Backscatter (dB)')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Panel 2: VH backscatter
        ax2 = axes[1]
        im2 = ax2.imshow(vh_db, cmap='gray', vmin=-30, vmax=-5)
        ax2.set_title('VH Backscatter (dB)')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Panel 3: Classification (this is the paintable panel)
        ax3 = axes[2]
        colors = ['#8B4513', '#FFFFFF', '#0066CC']  # Brown, White, Blue
        cmap = ListedColormap(colors)
        im3 = ax3.imshow(class_map, cmap=cmap, vmin=0, vmax=2)
        ax3.set_title('Classification (Paint Here!)', fontweight='bold')
        ax3.axis('off')
        
        legend_elements = [
            Patch(facecolor='#8B4513', label='0: Non-Glacier'),
            Patch(facecolor='#FFFFFF', edgecolor='black', label='1: Glacier'),
            Patch(facecolor='#0066CC', label='2: Glacial Lake')
        ]
        ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Painting context
        ctx = {
            'is_drawing': False,
            'current_class': None,
            'brush_radius': 12,
            'drawn_points': []
        }
        
        def paint(x, y, cls):
            r_center, c_center = int(y), int(x)
            radius = ctx['brush_radius']
            
            y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = x_grid**2 + y_grid**2 <= radius**2
            
            for r_offset, c_offset in zip(*np.where(mask)):
                r = r_center + r_offset - radius
                c = c_center + c_offset - radius
                
                if 0 <= r < class_map.shape[0] and 0 <= c < class_map.shape[1]:
                    USER_POINTS.append([r, c, cls])
                    ctx['drawn_points'].append((c, r, cls))
        
        def on_press(event):
            if event.inaxes != ax3 or event.xdata is None:
                return
            ctx['is_drawing'] = True
            
            if event.button == 1:
                ctx['current_class'] = 2  # Left -> Water
            elif event.button == 3:
                ctx['current_class'] = 0  # Right -> Non-glacier
            elif event.button == 2:
                ctx['current_class'] = 1  # Middle -> Glacier
            else:
                ctx['is_drawing'] = False
                return
            
            paint(event.xdata, event.ydata, ctx['current_class'])
        
        def on_move(event):
            if not ctx['is_drawing'] or event.inaxes != ax3 or event.xdata is None:
                return
            paint(event.xdata, event.ydata, ctx['current_class'])
        
        def on_release(event):
            if ctx['is_drawing'] and ctx['drawn_points']:
                # Update display
                updated_map = class_map.copy()
                for pt in ctx['drawn_points']:
                    c, r, cls = pt
                    if 0 <= r < class_map.shape[0] and 0 <= c < class_map.shape[1]:
                        updated_map[r, c] = cls
                im3.set_data(updated_map)
                fig.canvas.draw()
                print(f"  Painted {len(ctx['drawn_points'])} pixels as class {ctx['current_class']}")
            
            ctx['is_drawing'] = False
            ctx['drawn_points'] = []
        
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_move)
        fig.canvas.mpl_connect('button_release_event', on_release)
        
        plt.tight_layout()
        plt.show()
        
        return USER_POINTS
    
    def train_model(self, X_flat, shape, initial_map):
        """
        Train Random Forest using user corrections + sampled baseline pixels.
        """
        h, w = shape
        y_flat = initial_map.ravel()
        
        train_features = []
        train_labels = []
        
        # A. User correction points (high weight)
        if USER_POINTS:
            print(f"\n  [Learning] Adding {len(USER_POINTS)} manual corrections...")
            weight = 3
            for r, c, label in USER_POINTS:
                idx = r * w + c
                if 0 <= idx < len(X_flat):
                    feat = X_flat[idx]
                    for _ in range(weight):
                        train_features.append(feat)
                        train_labels.append(label)
        
        # B. Sample baseline pixels
        print("  [Learning] Sampling baseline structure...")
        num_samples = min(10000, len(X_flat))
        indices = np.random.choice(len(X_flat), num_samples, replace=False)
        for idx in indices:
            train_features.append(X_flat[idx])
            train_labels.append(y_flat[idx])
        
        train_features = np.array(train_features)
        train_labels = np.array(train_labels)
        
        # Scale features
        print("  [Learning] Scaling features...")
        self.scaler = StandardScaler()
        train_features_scaled = self.scaler.fit_transform(train_features)
        
        # Train RF
        print("  [Learning] Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        self.rf_model.fit(train_features_scaled, train_labels)
        
        # Predict on full image
        print("  [Learning] Applying model...")
        X_scaled = self.scaler.transform(X_flat)
        prediction = self.rf_model.predict(X_scaled)
        new_map = prediction.reshape(shape).astype(np.uint8)
        
        # Post-processing
        new_map = self.post_process_classification(new_map)
        
        # Save model
        joblib.dump(self.rf_model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"  [Memory] Model saved to {self.model_path}")
        
        return new_map
    
    def post_process_classification(self, class_map):
        """Apply morphological operations for cleaner results"""
        # Remove small isolated regions for each class
        for cls in [1, 2]:  # Glacier and Lake
            mask = (class_map == cls)
            cleaned = morphology.remove_small_objects(mask, min_size=100)
            cleaned = morphology.remove_small_holes(cleaned, area_threshold=50)
            class_map[mask & ~cleaned] = 0
            class_map[~mask & cleaned] = cls
        
        return class_map
    
    def save_vectors(self, class_map, profile, output_dir, date_str):
        """Export classification as GeoJSON vectors"""
        transform = profile['transform']
        crs = profile['crs']
        
        for class_val, class_name in [(1, 'Glacier'), (2, 'Lake')]:
            mask = (class_map == class_val).astype(np.uint8)
            if np.any(mask):
                results = (
                    {'properties': {'class': class_name, 'date': date_str, 'source': 'Sentinel-1'}, 'geometry': s}
                    for s, v in shapes(mask, mask=mask.astype(bool), transform=transform)
                )
                geoms = list(results)
                if geoms:
                    gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)
                    out_path = os.path.join(output_dir, f"S1_{class_name}_{date_str}.geojson")
                    gdf.to_file(out_path, driver='GeoJSON')
                    print(f"  [OK] Saved {class_name} vector: {out_path}")
    
    def create_visualization(self, class_map, vv_db, vh_db, date_str, output_path):
        """Generate publication-quality visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Sentinel-1 SAR Classification - {date_str}', fontsize=16, fontweight='bold')
        
        # Panel 1: VV
        ax1 = axes[0]
        im1 = ax1.imshow(vv_db, cmap='gray', vmin=-25, vmax=0)
        ax1.set_title('VV Backscatter (dB)')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Panel 2: VH
        ax2 = axes[1]
        im2 = ax2.imshow(vh_db, cmap='gray', vmin=-30, vmax=-5)
        ax2.set_title('VH Backscatter (dB)')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Panel 3: Classification
        ax3 = axes[2]
        colors = ['#8B4513', '#E8E8E8', '#0066CC']
        cmap = ListedColormap(colors)
        ax3.imshow(class_map, cmap=cmap, vmin=0, vmax=2)
        ax3.set_title('Classification', fontweight='bold')
        ax3.axis('off')
        
        legend_elements = [
            Patch(facecolor='#8B4513', label='Non-Glacier'),
            Patch(facecolor='#E8E8E8', edgecolor='black', label='Glacier'),
            Patch(facecolor='#0066CC', label='Glacial Lake')
        ]
        ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved visualization: {output_path}")
    
    def process_single(self, tiff_path, output_dir, interactive=True):
        """Process a single Sentinel-1 image"""
        filename = os.path.basename(tiff_path)
        date_str = filename.replace('.tif', '').replace('S1_', '')
        
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        
        # Load bands
        print("  Loading SAR bands...")
        vv, vh, angle, profile = self.load_sar_bands(tiff_path)
        
        # Compute features
        print("  Computing SAR features...")
        X_flat, shape, vv_db, vh_db = self.compute_sar_features(vv, vh, angle)
        
        # Compute texture for rule-based classifier
        _, local_std_vv, _, _ = self.compute_texture_features(vv_db)
        vv_vh_ratio = vv_db - vh_db
        
        # Initial classification
        print("  Generating initial classification...")
        if self.rf_model is not None and not interactive:
            # Use existing model
            X_scaled = self.scaler.transform(X_flat)
            prediction = self.rf_model.predict(X_scaled)
            class_map = prediction.reshape(shape).astype(np.uint8)
            class_map = self.post_process_classification(class_map)
        else:
            # Rule-based initial guess
            class_map = self.rule_based_classify(vv_db, vh_db, vv_vh_ratio, local_std_vv)
        
        # Interactive correction (if requested)
        if interactive:
            user_pts = self.interactive_correction(class_map, vv_db, vh_db, date_str)
            if user_pts:
                class_map = self.train_model(X_flat, shape, class_map)
            else:
                print("  No corrections made. Using initial classification.")
        
        # Save outputs
        os.makedirs(output_dir, exist_ok=True)
        self.create_visualization(class_map, vv_db, vh_db, date_str, 
                                  os.path.join(output_dir, f"S1_Result_{date_str}.png"))
        self.save_vectors(class_map, profile, output_dir, date_str)
        
        return class_map


def batch_process(tiff_dir, output_dir):
    """
    Process all Sentinel-1 images in a directory.
    First image: Interactive training
    Remaining images: Automatic batch using trained model
    """
    import glob
    
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "S1_*.tif")))
    
    if not tiff_files:
        print("No Sentinel-1 TIFF files found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nFound {len(tiff_files)} Sentinel-1 images")
    print("=" * 60)
    
    classifier = Sentinel1GlacierClassifier()
    
    # PHASE 1: Train on first image
    print("\n" + "=" * 60)
    print("PHASE 1: INTERACTIVE TRAINING")
    print("=" * 60)
    
    first_img = tiff_files[0]
    classifier.process_single(first_img, output_dir, interactive=True)
    
    # PHASE 2: Batch process remaining
    if len(tiff_files) > 1:
        print("\n" + "=" * 60)
        print(f"PHASE 2: BATCH PROCESSING ({len(tiff_files) - 1} images)")
        print("=" * 60)
        
        for tiff_path in tiff_files[1:]:
            classifier.process_single(tiff_path, output_dir, interactive=False)
    
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # Configuration
    TIFF_DIR = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_classification"
    
    batch_process(TIFF_DIR, OUTPUT_DIR)
