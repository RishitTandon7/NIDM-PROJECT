"""
Histogram-Enhanced Glacier Classifier
======================================
This classifier uses CLAHE (Contrast Limited Adaptive Histogram Equalization) 
preprocessing to dramatically improve classification accuracy.

Key Strategy:
1. Apply histogram equalization to enhance contrast
2. Use the enhanced bands for better feature separation
3. Apply physics-based rules on the enhanced data
4. Create final stitched mosaic

Author: Antigravity
Date: 2026-01-23
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.merge import merge
from scipy.ndimage import label, binary_opening, binary_closing
from skimage import exposure
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import warnings
import gc

warnings.filterwarnings('ignore')


class HistogramEnhancedClassifier:
    def __init__(self, tiff_dir, output_dir):
        self.tiff_dir = tiff_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "enhanced_bands"), exist_ok=True)
    
    def normalize_band(self, band, percentile_low=2, percentile_high=98):
        """
        Percentile-based normalization - robust to outliers
        """
        band = band.astype(np.float32)
        valid_mask = band > 0
        
        if not np.any(valid_mask):
            return np.zeros_like(band, dtype=np.float32)
        
        low = np.percentile(band[valid_mask], percentile_low)
        high = np.percentile(band[valid_mask], percentile_high)
        
        band_clipped = np.clip(band, low, high)
        band_normalized = (band_clipped - low) / (high - low + 1e-10)
        
        return band_normalized.astype(np.float32)
    
    def apply_clahe(self, band_normalized):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        This dramatically enhances local contrast
        """
        return exposure.equalize_adapthist(band_normalized, clip_limit=0.03).astype(np.float32)
    
    def load_and_enhance_bands(self, tiff_path):
        """
        Load bands and apply histogram equalization
        """
        try:
            with rasterio.open(tiff_path) as src:
                profile = src.profile
                
                # Load raw bands
                blue_raw = src.read(1).astype(np.float32)
                green_raw = src.read(2).astype(np.float32)
                red_raw = src.read(3).astype(np.float32)
                nir_raw = src.read(4).astype(np.float32)
                
                # SWIR
                if src.count >= 9:
                    swir_raw = src.read(9).astype(np.float32)
                elif src.count >= 5:
                    swir_raw = src.read(5).astype(np.float32)
                else:
                    swir_raw = np.zeros_like(blue_raw, dtype=np.float32)
            
            # Normalize each band
            blue_norm = self.normalize_band(blue_raw)
            green_norm = self.normalize_band(green_raw)
            red_norm = self.normalize_band(red_raw)
            nir_norm = self.normalize_band(nir_raw)
            swir_norm = self.normalize_band(swir_raw)
            
            # Apply CLAHE to each band
            blue_enh = self.apply_clahe(blue_norm)
            green_enh = self.apply_clahe(green_norm)
            red_enh = self.apply_clahe(red_norm)
            nir_enh = self.apply_clahe(nir_norm)
            swir_enh = self.apply_clahe(swir_norm)
            
            return {
                'blue': blue_enh,
                'green': green_enh,
                'red': red_enh,
                'nir': nir_enh,
                'swir': swir_enh,
                'profile': profile
            }
            
        except Exception as e:
            print(f"Error loading bands: {e}")
            return None
    
    def calculate_indices(self, bands):
        """
        Calculate spectral indices from enhanced bands
        """
        blue = bands['blue']
        green = bands['green']
        red = bands['red']
        nir = bands['nir']
        swir = bands['swir']
        
        eps = 1e-10
        
        # Water indices (enhanced bands make these more reliable)
        ndwi = (green - nir) / (green + nir + eps)
        mndwi = (green - swir) / (green + swir + eps)
        
        # Snow/Ice index
        ndsi = (green - swir) / (green + swir + eps)
        
        # Vegetation index
        ndvi = (nir - red) / (nir + red + eps)
        
        # Brightness (from enhanced bands)
        brightness = (blue + green + red) / 3.0
        
        return {
            'ndwi': np.nan_to_num(ndwi),
            'mndwi': np.nan_to_num(mndwi),
            'ndsi': np.nan_to_num(ndsi),
            'ndvi': np.nan_to_num(ndvi),
            'brightness': brightness
        }
    
    def estimate_water_depth(self, bands, water_mask):
        """
        Estimate relative water depth using spectral characteristics.
        
        Principle:
        - Deeper water absorbs more light (appears darker)
        - Blue light penetrates deeper than green/red
        - Blue/Green ratio indicates depth
        - NIR absorption is stronger in deeper water
        
        Returns depth categories: 0=No Water, 1=Shallow, 2=Medium, 3=Deep
        """
        blue = bands['blue']
        green = bands['green']
        nir = bands['nir']
        
        # Initialize depth map
        depth_map = np.zeros_like(blue, dtype=np.uint8)
        
        if not np.any(water_mask):
            return depth_map
        
        # Calculate depth indicators (only for water pixels)
        eps = 1e-10
        
        # Blue/Green ratio (higher = deeper water)
        bg_ratio = blue / (green + eps)
        
        # NIR absorption (lower NIR = deeper water)
        nir_absorption = 1.0 - nir
        
        # Brightness (darker = deeper)
        brightness = (blue + green) / 2.0
        darkness = 1.0 - brightness
        
        # Combine indicators for depth score
        # Weight: 40% darkness, 30% NIR absorption, 30% BG ratio
        depth_score = (
            0.4 * darkness +
            0.3 * nir_absorption +
            0.3 * np.clip(bg_ratio - 0.8, 0, 0.4) / 0.4  # Normalize BG ratio
        )
        
        # Classify depth (only where water exists)
        depth_map[water_mask & (depth_score < 0.3)] = 1   # Shallow
        depth_map[water_mask & (depth_score >= 0.3) & (depth_score < 0.5)] = 2  # Medium
        depth_map[water_mask & (depth_score >= 0.5)] = 3  # Deep
        
        return depth_map
    
    def classify_enhanced(self, bands, indices):
        """
        Classify using enhanced bands and indices
        
        With histogram equalization, the thresholds become more reliable:
        - Shadows: Very dark in ALL enhanced bands
        - Water: High NDWI, moderate brightness, low NIR
        - Glacier: Very bright, high NDSI, high NIR
        """
        blue = bands['blue']
        green = bands['green']
        nir = bands['nir']
        
        ndwi = indices['ndwi']
        mndwi = indices['mndwi']
        ndsi = indices['ndsi']
        ndvi = indices['ndvi']
        brightness = indices['brightness']
        
        h, w = blue.shape
        class_map = np.zeros((h, w), dtype=np.uint8)
        
        # ===== SHADOW DETECTION =====
        # After histogram equalization, shadows are VERY dark (< 0.1)
        # in all bands uniformly
        shadow_mask = (
            (brightness < 0.12) &
            (blue < 0.12) &
            (nir < 0.12) &
            (green < 0.12)
        )
        shadow_mask = binary_opening(shadow_mask, structure=np.ones((5, 5)))
        
        # ===== WATER DETECTION =====
        # Enhanced contrast makes water detection much more reliable
        # Water has:
        # - High NDWI (> 0.15)
        # - Low NIR (< 0.35) - water absorbs NIR
        # - Moderate brightness (0.15 - 0.6)
        # - Blue reflectance (> 0.15)
        water_mask = (
            (ndwi > 0.15) &
            (nir < 0.35) &
            (brightness > 0.15) &
            (brightness < 0.6) &
            (blue > 0.15) &
            (ndvi < 0.25) &
            (~shadow_mask)
        )
        
        # Clean water mask
        water_mask = binary_opening(water_mask, structure=np.ones((3, 3)))
        water_mask = self.remove_small_objects(water_mask, min_size=10)
        
        # ===== GLACIER DETECTION =====
        # Glaciers are very bright with high NDSI
        glacier_mask = (
            (brightness > 0.55) &
            (ndsi > 0.2) &
            (nir > 0.5) &
            (ndvi < 0.3) &
            (~shadow_mask) &
            (~water_mask)  # Don't overlap with water
        )
        
        # Clean glacier mask
        glacier_mask = binary_opening(glacier_mask, structure=np.ones((3, 3)))
        glacier_mask = binary_closing(glacier_mask, structure=np.ones((5, 5)))
        glacier_mask = self.remove_small_objects(glacier_mask, min_size=15)
        
        # Assign classes
        class_map[glacier_mask] = 1  # Glacier
        class_map[water_mask] = 2    # Water
        
        # Estimate water depth
        depth_map = self.estimate_water_depth(bands, water_mask)
        
        # Calculate statistics
        total_pixels = h * w
        water_pixels = np.sum(water_mask)
        
        depth_stats = {
            'shallow_pct': (np.sum(depth_map == 1) / total_pixels) * 100 if water_pixels > 0 else 0,
            'medium_pct': (np.sum(depth_map == 2) / total_pixels) * 100 if water_pixels > 0 else 0,
            'deep_pct': (np.sum(depth_map == 3) / total_pixels) * 100 if water_pixels > 0 else 0,
        }
        
        stats = {
            'shadow_pct': (np.sum(shadow_mask) / total_pixels) * 100,
            'glacier_pct': (np.sum(glacier_mask) / total_pixels) * 100,
            'water_pct': (np.sum(water_mask) / total_pixels) * 100,
            'depth_stats': depth_stats
        }
        
        return class_map, depth_map, stats
    
    def remove_small_objects(self, mask, min_size=10):
        """Remove small isolated regions"""
        labeled, num_features = label(mask)
        for i in range(1, num_features + 1):
            if np.sum(labeled == i) < min_size:
                mask[labeled == i] = False
        return mask
    
    def save_enhanced_preview(self, bands, date_str):
        """Save a preview of the enhanced RGB composite"""
        rgb = np.dstack([bands['red'], bands['green'], bands['blue']])
        rgb_clipped = np.clip(rgb, 0, 1)
        
        # Convert to uint8 for saving
        rgb_uint8 = (rgb_clipped * 255).astype(np.uint8)
        img = Image.fromarray(rgb_uint8)
        
        preview_path = os.path.join(self.output_dir, "enhanced_bands", f"Enhanced_{date_str}.png")
        img.save(preview_path)
        
        return preview_path
    
    def process_image(self, tiff_path):
        """Process a single image"""
        date_str = os.path.basename(tiff_path)[10:20]
        print(f"\nProcessing: {date_str}")
        
        # Load and enhance bands
        print("  Applying histogram equalization...")
        bands = self.load_and_enhance_bands(tiff_path)
        if bands is None:
            return None
        
        # Save enhanced preview
        self.save_enhanced_preview(bands, date_str)
        
        # Calculate indices
        print("  Calculating spectral indices...")
        indices = self.calculate_indices(bands)
        
        # Classify
        print("  Classifying...")
        class_map, depth_map, stats = self.classify_enhanced(bands, indices)
        
        depth_stats = stats['depth_stats']
        print(f"  Results: Glacier={stats['glacier_pct']:.2f}%, Water={stats['water_pct']:.2f}%, Shadow={stats['shadow_pct']:.2f}%")
        print(f"  Depth: Shallow={depth_stats['shallow_pct']:.2f}%, Medium={depth_stats['medium_pct']:.2f}%, Deep={depth_stats['deep_pct']:.2f}%")
        
        # Save outputs
        metadata = {
            'profile': bands['profile'],
            'date': date_str,
            'stats': stats
        }
        
        self.save_visualization(class_map, metadata)
        self.save_depth_visualization(depth_map, metadata)
        self.save_geotiff(class_map, metadata)
        self.save_depth_geotiff(depth_map, metadata)
        self.save_vector(class_map, metadata)
        
        return os.path.join(self.output_dir, f"Classified_{date_str}.tif")
    
    def save_visualization(self, class_map, metadata):
        """Save visualization"""
        date_str = metadata['date']
        stats = metadata['stats']
        
        colors = ['#8B4513', '#E8E8E8', '#00BFFF']  # Brown, Light Grey, Deep Sky Blue
        cmap = ListedColormap(colors)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(class_map, cmap=cmap, vmin=0, vmax=2)
        
        title = f'Histogram-Enhanced Classification: {date_str}\n'
        title += f'Glacier: {stats["glacier_pct"]:.2f}% | Water: {stats["water_pct"]:.2f}% | Shadow: {stats["shadow_pct"]:.2f}%'
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
        
        from matplotlib.patches import Patch
        legend = [
            Patch(facecolor='#8B4513', label='Non-Glacier'),
            Patch(facecolor='#E8E8E8', label='Glacier'),
            Patch(facecolor='#00BFFF', label='Glacial Lake')
        ]
        ax.legend(handles=legend, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, f"Classified_{date_str}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_depth_visualization(self, depth_map, metadata):
        """Save depth visualization"""
        date_str = metadata['date']
        depth_stats = metadata['stats']['depth_stats']
        
        # Color scheme: Black (No Water), Light Blue (Shallow), Blue (Medium), Dark Blue (Deep)
        colors = ['#000000', '#87CEEB', '#4682B4', '#00008B']
        cmap = ListedColormap(colors)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(depth_map, cmap=cmap, vmin=0, vmax=3)
        
        title = f'Water Depth Analysis: {date_str}\n'
        title += f'Shallow: {depth_stats["shallow_pct"]:.2f}% | Medium: {depth_stats["medium_pct"]:.2f}% | Deep: {depth_stats["deep_pct"]:.2f}%'
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
        
        from matplotlib.patches import Patch
        legend = [
            Patch(facecolor='#000000', label='No Water'),
            Patch(facecolor='#87CEEB', label='Shallow'),
            Patch(facecolor='#4682B4', label='Medium'),
            Patch(facecolor='#00008B', label='Deep')
        ]
        ax.legend(handles=legend, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, f"Depth_{date_str}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_depth_geotiff(self, depth_map, metadata):
        """Save depth map as GeoTIFF"""
        date_str = metadata['date']
        out_path = os.path.join(self.output_dir, f"Depth_{date_str}.tif")
        
        profile = metadata['profile'].copy()
        profile.update({
            'dtype': rasterio.uint8,
            'count': 1,
            'compress': 'lzw'
        })
        
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(depth_map, 1)
        
        return out_path
    
    def save_geotiff(self, class_map, metadata):
        """Save as GeoTIFF"""
        date_str = metadata['date']
        out_path = os.path.join(self.output_dir, f"Classified_{date_str}.tif")
        
        profile = metadata['profile'].copy()
        profile.update({
            'dtype': rasterio.uint8,
            'count': 1,
            'compress': 'lzw'
        })
        
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(class_map, 1)
        
        return out_path
    
    def save_vector(self, class_map, metadata):
        """Save as GeoJSON"""
        date_str = metadata['date']
        
        for class_val, class_name in [(1, 'Glacier'), (2, 'Lake')]:
            mask = (class_map == class_val)
            if np.any(mask):
                results = [
                    {'properties': {'class': class_name, 'date': date_str}, 'geometry': s}
                    for s, v in shapes(class_map, mask=mask, transform=metadata['profile']['transform'])
                ]
                if results:
                    gdf = gpd.GeoDataFrame.from_features(results, crs=metadata['profile']['crs'])
                    out_path = os.path.join(self.output_dir, f"{class_name}_{date_str}.geojson")
                    gdf.to_file(out_path, driver='GeoJSON')
    
    def process_all(self):
        """Process all images"""
        tiff_files = sorted(glob.glob(os.path.join(self.tiff_dir, "*.tif")))
        
        if not tiff_files:
            print("No TIF files found!")
            return []
        
        print(f"\n{'='*60}")
        print(f"HISTOGRAM-ENHANCED GLACIER CLASSIFIER")
        print(f"{'='*60}")
        print(f"Found {len(tiff_files)} images to process")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}")
        
        classified_paths = []
        
        for tiff_path in tiff_files:
            gc.collect()
            result = self.process_image(tiff_path)
            if result:
                classified_paths.append(result)
        
        return classified_paths
    
    def create_mosaic(self, classified_paths):
        """Create final stitched mosaic"""
        if not classified_paths:
            print("\nNo classified images to stitch!")
            return
        
        print(f"\n{'='*60}")
        print(f"Creating final mosaic from {len(classified_paths)} images...")
        print(f"{'='*60}")
        
        src_files = [rasterio.open(path) for path in classified_paths]
        mosaic, out_trans = merge(src_files)
        
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"
        })
        
        mosaic_path = os.path.join(self.output_dir, "Final_Histogram_Enhanced_Mosaic.tif")
        with rasterio.open(mosaic_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        for src in src_files:
            src.close()
        
        print(f"Mosaic saved: {mosaic_path}")
        
        # Create visualization
        self.visualize_mosaic(mosaic_path)
        
        return mosaic_path
    
    def visualize_mosaic(self, mosaic_path):
        """Create mosaic visualization"""
        with rasterio.open(mosaic_path) as src:
            data = src.read(1)
        
        colors = ['#8B4513', '#E8E8E8', '#00BFFF']
        cmap = ListedColormap(colors)
        
        plt.figure(figsize=(16, 12))
        plt.imshow(data, cmap=cmap, vmin=0, vmax=2)
        plt.title('Final Histogram-Enhanced Glacier Classification Mosaic', 
                  fontsize=16, fontweight='bold', pad=15)
        plt.axis('off')
        
        from matplotlib.patches import Patch
        legend = [
            Patch(facecolor='#8B4513', label='Non-Glacier'),
            Patch(facecolor='#E8E8E8', label='Glacier'),
            Patch(facecolor='#00BFFF', label='Glacial Lake')
        ]
        plt.legend(handles=legend, loc='upper right', fontsize=12)
        
        out_path = os.path.join(self.output_dir, "Final_Mosaic_Visualization.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Mosaic visualization saved: {out_path}")


if __name__ == "__main__":
    TIFF_DIR = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\histogram_enhanced_classification"
    
    classifier = HistogramEnhancedClassifier(TIFF_DIR, OUTPUT_DIR)
    classified_paths = classifier.process_all()
    
    if classified_paths:
        classifier.create_mosaic(classified_paths)
        print(f"\n{'='*60}")
        print("CLASSIFICATION COMPLETE!")
        print(f"Total images processed: {len(classified_paths)}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Check 'enhanced_bands' subfolder for histogram-equalized previews")
        print(f"{'='*60}\n")
    else:
        print("\nNo images were successfully processed.")
