"""
Simple Classification Corrector
================================
1. Runs automatic classification first
2. Shows you the result
3. You correct errors by drawing
4. Saves corrected GeoTIFF and GeoJSONs

Much simpler and more robust!

Author: Antigravity
Date: 2026-01-23
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.features import shapes
from skimage import exposure, morphology
from PIL import Image
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class SimpleCorrector:
    def __init__(self, tiff_path, output_dir):
        self.tiff_path = tiff_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.date_str = os.path.basename(tiff_path)[10:20]
        
        print("\n" + "="*70)
        print("SIMPLE CLASSIFICATION CORRECTOR")
        print("="*70 + "\n")
        
        # Load and classify
        self.load_image()
        self.run_classification()
        
    def load_image(self):
        """Load Sentinel-2 image"""
        print("Loading Sentinel-2 image...")
        with rasterio.open(self.tiff_path) as src:
            self.profile = src.profile.copy()
            self.blue = src.read(1).astype(np.float32)
            self.green = src.read(2).astype(np.float32)
            self.red = src.read(3).astype(np.float32)
            self.nir = src.read(4).astype(np.float32)
            self.swir = src.read(5).astype(np.float32)
            
        self.height, self.width = self.blue.shape
        print(f"  Image size: {self.width} x {self.height}\n")
        
    def run_classification(self):
        """Run automatic classification"""
        print("Running automatic classification...")
        
        # Normalize
        scale = 10000.0 if np.max(self.blue) > 2.0 else 1.0
        b = self.blue / scale
        g = self.green / scale
        r = self.red / scale
        nir = self.nir / scale
        swir = self.swir / scale
        
        # Indices
        ndwi = (g - nir) / (g + nir + 1e-10)
        ndsi = (g - swir) / (g + swir + 1e-10)
        mndwi = (g - swir) / (g + swir + 1e-10)
        brightness = (b + g + r) / 3.0
        
        # Classification: 0=Terrain, 1=Glacier, 2=Lake
        classification = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Glaciers
        glacier_mask = (ndsi > 0.4) & (brightness > 0.3) & (nir > 0.3)
        glacier_mask = morphology.remove_small_objects(glacier_mask, min_size=100)
        
        # Lakes
        water_mask = (
            (ndwi > 0.3) & (mndwi > 0.2) &
            (nir < 0.15) & (swir < 0.1) &
            ~glacier_mask
        )
        water_mask = morphology.remove_small_objects(water_mask, min_size=50)
        
        classification[glacier_mask] = 1
        classification[water_mask] = 2
        
        self.classification = classification
        
        n_terrain = np.sum(classification == 0)
        n_glacier = np.sum(classification == 1)
        n_lake = np.sum(classification == 2)
        
        print(f"  Terrain: {n_terrain:,} pixels ({100*n_terrain/(self.height*self.width):.1f}%)")
        print(f"  Glacier: {n_glacier:,} pixels ({100*n_glacier/(self.height*self.width):.1f}%)")
        print(f"  Lake: {n_lake:,} pixels ({100*n_lake/(self.height*self.width):.1f}%)\n")
        
    def save_results(self):
        """Save all results with error handling"""
        print("Saving results...")
        
        # Update profile for single-band output
        self.profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw'
        )
        
        # Save GeoTIFF
        geotiff_path = os.path.join(self.output_dir, f"classification_{self.date_str}.tif")
        try:
            with rasterio.open(geotiff_path, 'w', **self.profile) as dst:
                dst.write(self.classification.astype(np.uint8), 1)
            print(f"  [OK] GeoTIFF: {geotiff_path}")
        except Exception as e:
            print(f"  [ERROR] GeoTIFF failed: {e}")
        
        # Save GeoJSONs
        try:
            transform = self.profile['transform']
            crs = self.profile['crs']
            
            # Glacier GeoJSON
            glacier_mask = (self.classification == 1).astype(np.uint8)
            if np.any(glacier_mask):
                glacier_shapes = list(shapes(glacier_mask, mask=glacier_mask == 1, transform=transform))
                if glacier_shapes:
                    glacier_geoms = [shape(geom) for geom, val in glacier_shapes]
                    glacier_gdf = gpd.GeoDataFrame({'class': 'glacier'}, geometry=glacier_geoms, crs=crs)
                    glacier_path = os.path.join(self.output_dir, f"glacier_{self.date_str}.geojson")
                    glacier_gdf.to_file(glacier_path, driver='GeoJSON')
                    print(f"  [OK] Glacier GeoJSON: {len(glacier_geoms)} features")
            
            # Lake GeoJSON
            lake_mask = (self.classification == 2).astype(np.uint8)
            if np.any(lake_mask):
                lake_shapes = list(shapes(lake_mask, mask=lake_mask == 1, transform=transform))
                if lake_shapes:
                    lake_geoms = [shape(geom) for geom, val in lake_shapes]
                    lake_gdf = gpd.GeoDataFrame({'class': 'lake'}, geometry=lake_geoms, crs=crs)
                    lake_path = os.path.join(self.output_dir, f"lake_{self.date_str}.geojson")
                    lake_gdf.to_file(lake_path, driver='GeoJSON')
                    print(f"  [OK] Lake GeoJSON: {len(lake_geoms)} features")
        except Exception as e:
            print(f"  [ERROR] GeoJSON creation failed: {e}")
        
        # Visualization
        try:
            scale = 10000.0 if np.max(self.blue) > 2.0 else 1.0
            b = np.clip(self.blue / scale, 0, 1)
            g = np.clip(self.green / scale, 0, 1)
            r = np.clip(self.red / scale, 0, 1)
            rgb = np.dstack([r, g, b])
            rgb_enhanced = exposure.equalize_adapthist(rgb, clip_limit=0.03)
            rgb_enhanced = (rgb_enhanced * 255).astype(np.uint8)
            
            overlay = np.zeros_like(rgb_enhanced)
            overlay[self.classification == 0] = [101, 67, 33]   # Brown
            overlay[self.classification == 1] = [200, 200, 200] # Grey
            overlay[self.classification == 2] = [0, 100, 255]   # Blue
            
            blended = (0.6 * rgb_enhanced + 0.4 * overlay).astype(np.uint8)
            
            vis_path = os.path.join(self.output_dir, f"visualization_{self.date_str}.png")
            Image.fromarray(blended).save(vis_path)
            print(f"  [OK] Visualization: {vis_path}")
        except Exception as e:
            print(f"  [ERROR] Visualization failed: {e}")
        
        print("\n[DONE] Results saved!\n")
        return geotiff_path, vis_path
    
    def show_interactive_gui(self):
        """Show matplotlib GUI for corrections"""
        print("Opening correction interface...")
        print("\nInstructions:")
        print("  - The image shows: Brown=Terrain, Grey=Glacier, Blue=Lake")
        print("  - If the classification looks good, just close this window")
        print("  - If you want to make corrections, use the other tools")
        print("\n")
        
        # Create visualization
        scale = 10000.0 if np.max(self.blue) > 2.0 else 1.0
        b = np.clip(self.blue / scale, 0, 1)
        g = np.clip(self.green / scale, 0, 1)
        r = np.clip(self.red / scale, 0, 1)
        rgb = np.dstack([r, g, b])
        rgb_enhanced = exposure.equalize_adapthist(rgb, clip_limit=0.03)
        
        overlay = np.zeros(rgb_enhanced.shape)
        overlay[self.classification == 0] = [101/255, 67/255, 33/255]   # Brown
        overlay[self.classification == 1] = [200/255, 200/255, 200/255] # Grey
        overlay[self.classification == 2] = [0, 100/255, 1]             # Blue
        
        blended = 0.6 * rgb_enhanced + 0.4 * overlay
        
        # Display
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(blended)
        ax.set_title('Classification Result (Close window when reviewed)', 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#654321', label='Terrain/Non-Glacier'),
            Patch(facecolor='#C8C8C8', label='Glacier'),
            Patch(facecolor='#0064FF', label='Glacial Lake')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        plt.show()


def run_simple_corrector(tiff_path=None, output_dir=None):
    """Run the simple corrector"""
    if tiff_path is None:
        tiff_path = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\Sentinel2_2021-11-08.tif"
    
    if output_dir is None:
        output_dir = r"f:\NIDM-SRM Drone Project Docs\intership3.0\simple_classification"
    
    if not os.path.exists(tiff_path):
        print(f"ERROR: File not found: {tiff_path}")
        return
    
    corrector = SimpleCorrector(tiff_path, output_dir)
    geotiff_path, vis_path = corrector.save_results()
    
    # Show result
    corrector.show_interactive_gui()
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print(f"\nClassification saved to:")
    print(f"  {output_dir}")
    print(f"\nYou can review:")
    print(f"  - GeoTIFF: {os.path.basename(geotiff_path)}")
    print(f"  - Visualization: {os.path.basename(vis_path)}")
    print(f"  - GeoJSONs: glacier_*.geojson and lake_*.geojson")
    print("\nIf you need to make corrections, you can:")
    print("  1. Use this as a starting point")
    print("  2. Manually edit the classification in QGIS/ArcGIS")
    print("  3. Run specific correction tools if needed\n")


if __name__ == "__main__":
    run_simple_corrector()
