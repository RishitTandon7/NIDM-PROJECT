"""
Adaptive Glacier and Glacial Lake Classifier
Optimized for real Sentinel-2 data with better lake detection

Key improvements:
1. Relaxed water detection thresholds
2. Uses relative brightness instead of absolute values
3. Better handling of turbid/sediment-filled glacial lakes
4. Visual debugging output to tune parameters
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy.ndimage import label, binary_opening, binary_closing, percentile_filter
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
import gc

warnings.filterwarnings('ignore')

class AdaptiveGlacierClassifier:
    def __init__(self, tiff_dir, output_dir, dem_path=None):
        self.tiff_dir = tiff_dir
        self.output_dir = output_dir
        self.dem_path = dem_path
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_bands(self, src):
        """Load Sentinel-2 bands"""
        try:
            blue = src.read(1).astype(np.float32)
            green = src.read(2).astype(np.float32)
            red = src.read(3).astype(np.float32)
            nir = src.read(4).astype(np.float32)
            
            if src.count >= 9:
                swir = src.read(9).astype(np.float32)
            elif src.count >= 5:
                swir = src.read(5).astype(np.float32)
            else:
                swir = np.zeros_like(blue, dtype=np.float32)
                
            return blue, green, red, nir, swir
        except Exception as e:
            print(f"Error loading bands: {e}")
            return None
    
    def normalize_bands(self, blue, green, red, nir, swir):
        """Normalize bands to 0-1 range for adaptive thresholding"""
        bands = [blue, green, red, nir, swir]
        normalized = []
        
        for band in bands:
            # Use percentile normalization to handle outliers
            p2, p98 = np.percentile(band[band > 0], [2, 98])
            band_norm = np.clip((band - p2) / (p98 - p2 + 1e-10), 0, 1)
            normalized.append(band_norm)
        
        return normalized
    
    def calculate_indices(self, blue, green, red, nir, swir):
        """Calculate spectral indices"""
        eps = 1e-10
        
        # Normalize for adaptive processing
        blue_n, green_n, red_n, nir_n, swir_n = self.normalize_bands(blue, green, red, nir, swir)
        
        # Water indices
        ndwi = (green - nir) / (green + nir + eps)
        mndwi = (green - swir) / (green + swir + eps)
        
        # Alternative water index using normalized bands
        aweinsh = blue + 2.5 * green - 1.5 * (nir + swir)
        
        # Snow/Ice index
        ndsi = (green - swir) / (green + swir + eps)
        
        # Vegetation index
        ndvi = (nir - red) / (nir + red + eps)
        
        # Brightness metrics
        brightness = (blue + green + red) / 3.0
        brightness_norm = (blue_n + green_n + red_n) / 3.0
        
        # Turbidity index (for sediment-filled glacial lakes)
        turbidity = (red + green) / (blue + eps)
        
        return {
            'ndwi': np.nan_to_num(ndwi),
            'mndwi': np.nan_to_num(mndwi),
            'aweinsh': np.nan_to_num(aweinsh),
            'ndsi': np.nan_to_num(ndsi),
            'ndvi': np.nan_to_num(ndvi),
            'brightness': brightness,
            'brightness_norm': brightness_norm,
            'turbidity': np.nan_to_num(turbidity),
            'blue_n': blue_n,
            'green_n': green_n,
            'nir_n': nir_n
        }
    
    def detect_shadows_adaptive(self, blue, green, red, nir, indices):
        """
        Adaptive shadow detection using relative brightness
        Shadows are uniformly dark across ALL bands
        """
        brightness_norm = indices['brightness_norm']
        blue_n = indices['blue_n']
        nir_n = indices['nir_n']
        
        # Calculate local brightness variation
        local_mean = percentile_filter(brightness_norm, 50, size=15)
        
        # Shadows are:
        # 1. Very dark compared to surroundings
        # 2. Dark in all bands uniformly
        # 3. Low absolute brightness
        shadow_mask = (
            (brightness_norm < 0.15) &  # Very low normalized brightness
            (blue_n < 0.15) &
            (nir_n < 0.15) &
            (brightness_norm < local_mean * 0.5)  # Much darker than surroundings
        )
        
        # Morphological cleaning
        shadow_mask = binary_opening(shadow_mask, structure=np.ones((7, 7)))
        
        return shadow_mask
    
    def detect_water_adaptive(self, blue, green, red, nir, swir, indices, shadow_mask):
        """
        Adaptive water detection optimized for glacial lakes
        Handles both clear and turbid (sediment-filled) lakes
        """
        ndwi = indices['ndwi']
        mndwi = indices['mndwi']
        aweinsh = indices['aweinsh']
        ndvi = indices['ndvi']
        brightness_norm = indices['brightness_norm']
        turbidity = indices['turbidity']
        blue_n = indices['blue_n']
        green_n = indices['green_n']
        nir_n = indices['nir_n']
        
        # TIER 1: Clear glacial lakes (high NDWI, blue-green water)
        clear_water = (
            (ndwi > 0.2) &  # Relaxed from 0.35
            (mndwi > 0.15) &  # Relaxed from 0.3
            (nir_n < 0.4) &  # Water absorbs NIR
            (brightness_norm > 0.2) &  # Not too dark (not shadow)
            (brightness_norm < 0.7) &  # Not too bright (not ice)
            (ndvi < 0.3) &  # Not vegetation
            (~shadow_mask)
        )
        
        # TIER 2: Turbid/sediment-filled glacial lakes
        # These appear greenish-brown due to glacial flour
        turbid_water = (
            (ndwi > 0.1) &  # Still has water signature but weaker
            (aweinsh > -0.5) &  # Alternative water index
            (nir_n < 0.5) &  # NIR absorption
            (brightness_norm > 0.25) &
            (brightness_norm < 0.65) &
            (turbidity > 0.8) &  # High turbidity
            (turbidity < 1.5) &
            (green_n > 0.3) &  # Some green reflectance
            (ndvi < 0.35) &
            (~shadow_mask)
        )
        
        # TIER 3: Very dark water (deep lakes)
        dark_water = (
            (ndwi > 0.25) &
            (mndwi > 0.2) &
            (nir_n < 0.3) &  # Strong NIR absorption
            (brightness_norm > 0.15) &  # Just above shadow threshold
            (brightness_norm < 0.45) &
            (blue_n > 0.15) &  # Has some blue reflectance
            (ndvi < 0.25) &
            (~shadow_mask)
        )
        
        # Combine all water types
        water_mask = clear_water | turbid_water | dark_water
        
        # Spatial filtering - remove very small features
        water_mask = binary_opening(water_mask, structure=np.ones((3, 3)))
        water_mask = self.remove_small_objects(water_mask, min_size=8)
        
        return water_mask
    
    def detect_glacier_adaptive(self, blue, green, red, nir, swir, indices, shadow_mask):
        """
        Adaptive glacier/snow detection
        """
        ndsi = indices['ndsi']
        ndvi = indices['ndvi']
        brightness_norm = indices['brightness_norm']
        nir_n = indices['nir_n']
        
        # Glaciers are bright with high NDSI and NIR reflectance
        glacier_mask = (
            (brightness_norm > 0.6) &  # Very bright
            (ndsi > 0.25) &  # Relaxed from 0.35
            (nir_n > 0.6) &  # High NIR reflectance
            (ndvi < 0.3) &  # Not vegetation
            (~shadow_mask)
        )
        
        # Spatial filtering
        glacier_mask = binary_opening(glacier_mask, structure=np.ones((3, 3)))
        glacier_mask = binary_closing(glacier_mask, structure=np.ones((5, 5)))
        glacier_mask = self.remove_small_objects(glacier_mask, min_size=15)
        
        return glacier_mask
    
    def remove_small_objects(self, mask, min_size=10):
        """Remove small isolated regions"""
        labeled, num_features = label(mask)
        for i in range(1, num_features + 1):
            if np.sum(labeled == i) < min_size:
                mask[labeled == i] = False
        return mask
    
    def classify_image(self, tiff_path):
        """Classify a single Sentinel-2 image"""
        print(f"\nProcessing: {os.path.basename(tiff_path)}")
        
        with rasterio.open(tiff_path) as src:
            profile = src.profile
            date_str = os.path.basename(tiff_path)[10:20]
            
            bands = self.load_bands(src)
            if bands is None:
                return None, None
            
            blue, green, red, nir, swir = bands
            
            # Print band statistics for debugging
            print(f"  Band ranges:")
            print(f"    Blue: {blue.min():.0f} - {blue.max():.0f}")
            print(f"    Green: {green.min():.0f} - {green.max():.0f}")
            print(f"    NIR: {nir.min():.0f} - {nir.max():.0f}")
            print(f"    SWIR: {swir.min():.0f} - {swir.max():.0f}")
            
            # Calculate indices
            indices = self.calculate_indices(blue, green, red, nir, swir)
            
            # Print index statistics
            print(f"  Index ranges:")
            print(f"    NDWI: {indices['ndwi'].min():.3f} - {indices['ndwi'].max():.3f}")
            print(f"    MNDWI: {indices['mndwi'].min():.3f} - {indices['mndwi'].max():.3f}")
            print(f"    NDSI: {indices['ndsi'].min():.3f} - {indices['ndsi'].max():.3f}")
            
            # Initialize classification
            h, w = blue.shape
            class_map = np.zeros((h, w), dtype=np.uint8)
            
            # Detect features
            print("  Detecting shadows...")
            shadow_mask = self.detect_shadows_adaptive(blue, green, red, nir, indices)
            shadow_pct = (np.sum(shadow_mask) / shadow_mask.size) * 100
            print(f"    Shadow: {shadow_pct:.2f}%")
            
            print("  Detecting water bodies...")
            water_mask = self.detect_water_adaptive(blue, green, red, nir, swir, indices, shadow_mask)
            water_pct = (np.sum(water_mask) / water_mask.size) * 100
            print(f"    Water: {water_pct:.2f}%")
            
            print("  Detecting glaciers...")
            glacier_mask = self.detect_glacier_adaptive(blue, green, red, nir, swir, indices, shadow_mask)
            glacier_pct = (np.sum(glacier_mask) / glacier_mask.size) * 100
            print(f"    Glacier: {glacier_pct:.2f}%")
            
            # Assign classes
            class_map[glacier_mask] = 1
            class_map[water_mask] = 2
            
            # Save intermediate results for debugging
            self.save_debug_indices(indices, date_str, water_mask, glacier_mask, shadow_mask)
            
            metadata = {
                'profile': profile,
                'date': date_str,
                'transform': src.transform,
                'crs': src.crs,
                'shape': (h, w),
                'stats': {
                    'shadow_pct': shadow_pct,
                    'water_pct': water_pct,
                    'glacier_pct': glacier_pct
                }
            }
            
            return class_map, metadata
    
    def save_debug_indices(self, indices, date_str, water_mask, glacier_mask, shadow_mask):
        """Save debug visualization of indices and masks"""
        debug_dir = os.path.join(self.output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Debug Visualization: {date_str}', fontsize=14, fontweight='bold')
        
        # NDWI
        im1 = axes[0, 0].imshow(indices['ndwi'], cmap='RdYlBu', vmin=-0.5, vmax=0.5)
        axes[0, 0].set_title('NDWI (Water Index)')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        # MNDWI
        im2 = axes[0, 1].imshow(indices['mndwi'], cmap='RdYlBu', vmin=-0.5, vmax=0.5)
        axes[0, 1].set_title('MNDWI')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        
        # NDSI
        im3 = axes[0, 2].imshow(indices['ndsi'], cmap='RdYlGn', vmin=-0.5, vmax=0.5)
        axes[0, 2].set_title('NDSI (Snow Index)')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
        
        # Brightness
        im4 = axes[0, 3].imshow(indices['brightness_norm'], cmap='gray', vmin=0, vmax=1)
        axes[0, 3].set_title('Normalized Brightness')
        axes[0, 3].axis('off')
        plt.colorbar(im4, ax=axes[0, 3], fraction=0.046)
        
        # Shadow mask
        axes[1, 0].imshow(shadow_mask, cmap='gray')
        axes[1, 0].set_title('Shadow Mask')
        axes[1, 0].axis('off')
        
        # Water mask
        axes[1, 1].imshow(water_mask, cmap='Blues')
        axes[1, 1].set_title('Water Mask')
        axes[1, 1].axis('off')
        
        # Glacier mask
        axes[1, 2].imshow(glacier_mask, cmap='gray')
        axes[1, 2].set_title('Glacier Mask')
        axes[1, 2].axis('off')
        
        # Combined
        combined = np.zeros_like(shadow_mask, dtype=np.uint8)
        combined[glacier_mask] = 1
        combined[water_mask] = 2
        colors = ['#8B4513', '#D3D3D3', '#1E90FF']
        cmap = ListedColormap(colors)
        axes[1, 3].imshow(combined, cmap=cmap, vmin=0, vmax=2)
        axes[1, 3].set_title('Final Classification')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        debug_path = os.path.join(debug_dir, f"Debug_{date_str}.png")
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Debug visualization saved: {os.path.basename(debug_path)}")
    
    def save_visualization(self, class_map, metadata):
        """Save classified image as PNG"""
        date_str = metadata['date']
        out_path = os.path.join(self.output_dir, f"Classified_{date_str}.png")
        
        colors = ['#8B4513', '#D3D3D3', '#1E90FF']
        cmap = ListedColormap(colors)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(class_map, cmap=cmap, vmin=0, vmax=2)
        
        stats = metadata['stats']
        title = f'Glacier Classification: {date_str}\n'
        title += f'Glacier: {stats["glacier_pct"]:.2f}% | Water: {stats["water_pct"]:.2f}% | Shadow: {stats["shadow_pct"]:.2f}%'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#8B4513', label='Non-Glacier'),
            Patch(facecolor='#D3D3D3', label='Glacier'),
            Patch(facecolor='#1E90FF', label='Glacial Lake')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved visualization: {os.path.basename(out_path)}")
    
    def save_vector(self, class_map, metadata):
        """Save classification as GeoJSON"""
        date_str = metadata['date']
        
        for class_val, class_name in [(1, 'Glacier'), (2, 'Lake')]:
            mask = (class_map == class_val)
            if np.any(mask):
                results = [
                    {'properties': {'class': class_name, 'date': date_str}, 'geometry': s}
                    for s, v in shapes(class_map, mask=mask, transform=metadata['transform'])
                ]
                if results:
                    gdf = gpd.GeoDataFrame.from_features(results, crs=metadata['crs'])
                    out_path = os.path.join(self.output_dir, f"{class_name}_{date_str}.geojson")
                    gdf.to_file(out_path, driver='GeoJSON')
                    print(f"  Saved {class_name.lower()} vector: {os.path.basename(out_path)}")
    
    def save_geotiff(self, class_map, metadata):
        """Save classification as GeoTIFF"""
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
        
        print(f"  Saved GeoTIFF: {os.path.basename(out_path)}")
        return out_path
    
    def process_all(self):
        """Process all Sentinel-2 images"""
        tiff_files = sorted(glob.glob(os.path.join(self.tiff_dir, "*.tif")))
        
        if not tiff_files:
            print("No TIF files found!")
            return []
        
        print(f"Found {len(tiff_files)} images to process")
        
        classified_paths = []
        
        for tiff_path in tiff_files:
            gc.collect()
            
            class_map, metadata = self.classify_image(tiff_path)
            
            if class_map is None:
                continue
            
            self.save_visualization(class_map, metadata)
            self.save_vector(class_map, metadata)
            geotiff_path = self.save_geotiff(class_map, metadata)
            classified_paths.append(geotiff_path)
        
        return classified_paths
    
    def create_mosaic(self, classified_paths):
        """Create final mosaic"""
        if not classified_paths:
            print("No classified images to stitch!")
            return
        
        print(f"\nStitching {len(classified_paths)} images...")
        
        from rasterio.merge import merge
        
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
        
        mosaic_path = os.path.join(self.output_dir, "Final_Glacier_Mosaic.tif")
        with rasterio.open(mosaic_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        for src in src_files:
            src.close()
        
        print(f"Final mosaic saved: {mosaic_path}")
        self.visualize_mosaic(mosaic_path)
        
        return mosaic_path
    
    def visualize_mosaic(self, mosaic_path):
        """Visualize final mosaic"""
        with rasterio.open(mosaic_path) as src:
            data = src.read(1)
        
        colors = ['#8B4513', '#D3D3D3', '#1E90FF']
        cmap = ListedColormap(colors)
        
        plt.figure(figsize=(16, 12))
        plt.imshow(data, cmap=cmap, vmin=0, vmax=2)
        plt.title('Final Glacier Classification Mosaic', fontsize=16, fontweight='bold')
        plt.axis('off')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#8B4513', label='Non-Glacier'),
            Patch(facecolor='#D3D3D3', label='Glacier'),
            Patch(facecolor='#1E90FF', label='Glacial Lake')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        out_path = os.path.join(self.output_dir, "Final_Mosaic_Visualization.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Mosaic visualization saved: {out_path}")

if __name__ == "__main__":
    TIFF_DIR = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\adaptive_classification"
    
    classifier = AdaptiveGlacierClassifier(TIFF_DIR, OUTPUT_DIR)
    classified_paths = classifier.process_all()
    
    if classified_paths:
        classifier.create_mosaic(classified_paths)
        print("\n" + "="*60)
        print("ADAPTIVE CLASSIFICATION COMPLETE!")
        print(f"Total images processed: {len(classified_paths)}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Check the 'debug' subfolder for detailed index visualizations!")
        print("="*60)
    else:
        print("\nNo images were successfully processed.")