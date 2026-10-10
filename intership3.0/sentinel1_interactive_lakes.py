"""
Interactive Lake Selection Tool
================================
Simple tool to manually select lake regions:
1. View the SAR image
2. Click and drag to draw rectangles around lakes
3. Press 'S' to save, 'R' to reset, 'Q' to quit

This bypasses the shadow problem entirely - YOU decide what's a lake.

Author: Antigravity
Date: 2026-01-30
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
from PIL import Image
import geopandas as gpd
from scipy.ndimage import uniform_filter, median_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector, Button
import warnings

warnings.filterwarnings('ignore')


class LakeSelector:
    """Interactive lake selection tool"""
    
    def __init__(self, vv, vh, profile, output_dir, date_str):
        self.vv = vv
        self.vh = vh
        self.profile = profile
        self.output_dir = output_dir
        self.date_str = date_str
        self.h, self.w = vv.shape
        
        # Lake mask
        self.lake_mask = np.zeros((self.h, self.w), dtype=bool)
        self.rectangles = []
        
        # Prepare display image
        p1, p99 = np.percentile(vv, [1, 99])
        self.display_img = np.clip((vv - p1) / (p99 - p1), 0, 1)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def on_select(self, eclick, erelease):
        """Handle rectangle selection"""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Ensure proper order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Clip to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.w, x2)
        y2 = min(self.h, y2)
        
        # Add to mask
        self.lake_mask[y1:y2, x1:x2] = True
        self.rectangles.append((x1, y1, x2-x1, y2-y1))
        
        # Update display
        self.update_display()
        
        lake_pct = self.lake_mask.sum() / self.lake_mask.size * 100
        print(f"  Added region: ({x1},{y1}) to ({x2},{y2}). Total lake: {lake_pct:.2f}%")
    
    def update_display(self):
        """Update the display with current selections"""
        # Create RGB display
        rgb = np.stack([self.display_img, self.display_img, self.display_img], axis=-1)
        
        # Highlight lakes in blue
        rgb[self.lake_mask, 0] = 0
        rgb[self.lake_mask, 1] = 0.3
        rgb[self.lake_mask, 2] = 0.8
        
        self.ax.imshow(rgb)
        self.fig.canvas.draw()
    
    def save_results(self, event=None):
        """Save the lake mask"""
        lake_pct = self.lake_mask.sum() / self.lake_mask.size * 100
        print(f"\n  Saving results... ({lake_pct:.2f}% lake)")
        
        # Create classification map (0=other, 2=lake)
        class_map = np.zeros((self.h, self.w), dtype=np.uint8)
        class_map[self.lake_mask] = 2
        
        # Save GeoTIFF
        tiff_path = os.path.join(self.output_dir, f"S1_Lake_Manual_{self.date_str}.tif")
        profile = self.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
        with rasterio.open(tiff_path, 'w', **profile) as dst:
            dst.write(class_map, 1)
        print(f"  [OK] Saved GeoTIFF: {tiff_path}")
        
        # Save PNG (lake overlay on SAR)
        png_path = os.path.join(self.output_dir, f"S1_Lake_Manual_{self.date_str}.png")
        rgb = np.stack([self.display_img, self.display_img, self.display_img], axis=-1)
        rgb[self.lake_mask, 0] = 0
        rgb[self.lake_mask, 1] = 0.2
        rgb[self.lake_mask, 2] = 0.9
        rgb = (rgb * 255).astype(np.uint8)
        Image.fromarray(rgb).save(png_path)
        print(f"  [OK] Saved PNG: {png_path}")
        
        # Save GeoJSON
        if np.any(self.lake_mask):
            mask_uint8 = self.lake_mask.astype(np.uint8)
            transform = self.profile['transform']
            crs = self.profile['crs']
            
            results = list(shapes(mask_uint8, mask=self.lake_mask, transform=transform))
            if results:
                geoms = [{'properties': {'class': 'Lake', 'date': self.date_str, 'method': 'manual'}, 
                          'geometry': s} for s, v in results]
                gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)
                geojson_path = os.path.join(self.output_dir, f"S1_Lake_Manual_{self.date_str}.geojson")
                gdf.to_file(geojson_path, driver='GeoJSON')
                print(f"  [OK] Saved GeoJSON: {geojson_path}")
        
        print("\n  Results saved successfully!")
    
    def reset(self, event=None):
        """Reset all selections"""
        self.lake_mask = np.zeros((self.h, self.w), dtype=bool)
        self.rectangles = []
        self.update_display()
        print("  Reset all selections")
    
    def run(self):
        """Run the interactive selector"""
        print("\n" + "=" * 60)
        print("INTERACTIVE LAKE SELECTION TOOL")
        print("=" * 60)
        print("\nInstructions:")
        print("  - Click and drag to draw rectangles around lakes")
        print("  - Press 'S' or click Save button to save results")
        print("  - Press 'R' or click Reset button to clear")
        print("  - Press 'Q' or close window to quit")
        print("=" * 60 + "\n")
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.ax.imshow(self.display_img, cmap='gray')
        self.ax.set_title(f"Draw rectangles around LAKES - {self.date_str}\n(Drag to select, S=Save, R=Reset, Q=Quit)")
        
        # Rectangle selector
        self.selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            button=[1],  # Left click
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False
        )
        
        # Buttons
        ax_save = plt.axes([0.7, 0.02, 0.1, 0.04])
        ax_reset = plt.axes([0.81, 0.02, 0.1, 0.04])
        
        btn_save = Button(ax_save, 'Save')
        btn_reset = Button(ax_reset, 'Reset')
        
        btn_save.on_clicked(self.save_results)
        btn_reset.on_clicked(self.reset)
        
        # Keyboard shortcuts
        def on_key(event):
            if event.key == 's':
                self.save_results()
            elif event.key == 'r':
                self.reset()
            elif event.key == 'q':
                plt.close()
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        plt.tight_layout()
        plt.show()


def run_interactive_selection(tiff_path, output_dir):
    """Run interactive lake selection"""
    filename = os.path.basename(tiff_path)
    date_str = filename.replace('.tif', '').replace('S1_', '')
    
    print("Loading SAR data...")
    with rasterio.open(tiff_path) as src:
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)
        profile = src.profile.copy()
    
    # Denoise for better display
    vv = median_filter(vv, size=3)
    local_mean = uniform_filter(vv.astype(np.float64), 5)
    local_sqr_mean = uniform_filter(vv.astype(np.float64)**2, 5)
    local_var = np.maximum(local_sqr_mean - local_mean**2, 0)
    overall_var = np.var(vv)
    w = local_var / (local_var + overall_var + 1e-10)
    vv = (local_mean + w * (vv - local_mean)).astype(np.float32)
    
    selector = LakeSelector(vv, vh, profile, output_dir, date_str)
    selector.run()


if __name__ == "__main__":
    TIFF_PATH = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1\S1_2023-07-18.tif"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_manual_lakes"
    
    run_interactive_selection(TIFF_PATH, OUTPUT_DIR)
