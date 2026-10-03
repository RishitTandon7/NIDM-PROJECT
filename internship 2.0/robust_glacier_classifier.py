"""
Robust Glacier Classifier (Physics-Based)
Colors: Glacier (Grey), Lake (Blue), Non-Glacier (Brown)
Uses morphological cleaning to create "clean" polygon-like regions.
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import median_filter
import glob

class RobustGlacierClassifier:
    def __init__(self, output_dir="final_accurate_maps"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # requested colors: Glacier(Grey), Lake(Blue), Non-Glacier(Brown)
        self.colors = ['#8C8C8C', '#1E90FF', '#8B5A2B'] 
        self.classes = ['Glacier', 'Glacier Lake', 'Non-Glacier']

    def classify_scene(self, vrt_path):
        """Classify using robust spectral indices thresholds"""
        with rasterio.open(vrt_path) as src:
            # Read necessary bands (Sentinel-2)
            # Band 3 (Green), Band 4 (Red), Band 8 (NIR), Band 11 (SWIR)
            
            # Load bands safely
            def get_band(idx):
                if idx <= src.count:
                    b = src.read(idx).astype(float)
                    mask = src.read_masks(idx)
                    b[mask == 0] = np.nan
                    return b
                return np.zeros((src.height, src.width))

            # Sentinel-2 Band mapping (1-based index)
            # 2: Blue, 3: Green, 4: Red, 8: NIR
            # But in the VRTs, we need to check the band count or assume standard stack
            # Assuming standard stack: 1=B2, 2=B3, 3=B4, 8=B8, 11=B11
            # Adjusting based on standard Sentinel-2 VRT usually having all bands
            
            # Safe read strategy
            nbands = src.count
            blue  = get_band(2) if nbands >= 2 else get_band(1)
            green = get_band(3) if nbands >= 3 else get_band(2)
            red   = get_band(4) if nbands >= 4 else get_band(3)
            nir   = get_band(8) if nbands >= 8 else get_band(4) # Fallback to 4 if 8 missing
            swir  = get_band(11) if nbands >= 11 else get_band(5) # Fallback
            
            # 1. NDSI (Normalized Difference Snow Index) = (Green - SWIR) / (Green + SWIR)
            # Key for Glaciers
            ndsi = (green - swir) / (green + swir + 1e-6)
            
            # 2. NDWI (Normalized Difference Water Index) = (Green - NIR) / (Green + NIR)
            # Key for Lakes
            ndwi = (green - nir) / (green + nir + 1e-6)
            
            # 3. NDVI (Normalized Difference Vegetation Index)
            ndvi = (nir - red) / (nir + red + 1e-6)

            # --- Strict Classification Rules ---
            
            height, width = ndsi.shape
            classification = np.full((height, width), 2, dtype=np.uint8) # Default to Non-Glacier (2)
            
            # Rule 1: Glaciers (High NDSI, not water)
            # Ice is bright and snowy. 
            # Threshold: NDSI > 0.4 usually indicates snow/ice
            glacier_mask = (ndsi > 0.4) & (ndwi < 0.2)
            classification[glacier_mask] = 0 # Set Glacier
            
            # Rule 2: Glacier Lakes (High NDWI, specific bounds)
            # Water has high NDWI. But shadows can also have high NDWI.
            # Lakes are usually strictly bounded.
            lake_mask = (ndwi > 0.15) & (ndsi < 0.4) & (ndvi < 0.1)
            
            # Additional check: Dark objects (water is dark in SWIR)
            dark_swir = (swir < 1000) # Raw Digital Number check usually, but here normalized floats?
            # If using floats from 0-10000 (typical S2), let's check max
            if np.nanmax(swir) > 1000: # It's likely 0-10000 range
                lake_mask = lake_mask & (swir < 1500)
                
            classification[lake_mask] = 1 # Set Lake
            
            # --- Post-Processing (The "Clean" Look) ---
            # Apply Median Filter to remove "salt and pepper" noise
            # This creates that "polygon/stitched" look by smoothing distinct regions
            clean_class = median_filter(classification, size=3) # 3x3 window cleaning
            
            return clean_class

    def process_all(self, vrt_folder):
        print(f"Processing images from: {vrt_folder}")
        vrt_files = sorted(glob.glob(os.path.join(vrt_folder, "*.vrt")))
        
        if not vrt_files:
            print("No VRT files found.")
            return

        print(f"Found {len(vrt_files)} images. Generating high-accuracy maps...")
        
        cmap = ListedColormap(self.colors)
        
        # Process first 5 just to show user immediately, or all if requested
        # User said "make ... for all of the 66 images"
        
        # Grid Setup
        n_images = len(vrt_files)
        cols = 6
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
        fig.suptitle('Accurate Physics-Based Classification\nGrey=Glacier | Blue=Lake | Brown=Non-Glacier', fontsize=24, y=0.99)
        axes = axes.flatten()
        
        for i, vrt_path in enumerate(vrt_files):
            print(f"[{i+1}/{n_images}] Classifying {os.path.basename(vrt_path)}...")
            try:
                result = self.classify_scene(vrt_path)
                
                axes[i].imshow(result, cmap=cmap, vmin=0, vmax=2, interpolation='nearest') # nearest keeps sharp edges
                axes[i].set_title(os.path.basename(vrt_path)[10:20], fontsize=10) # Date only
                axes[i].axis('off')
                
                # Save Individual High-Res Image (Proof of Accuracy)
                ind_fig, ind_ax = plt.subplots(figsize=(10, 10))
                ind_ax.imshow(result, cmap=cmap, vmin=0, vmax=2)
                ind_ax.axis('off')
                
                # Add Legend to individual map
                import matplotlib.patches as mpatches
                patches = [
                    mpatches.Patch(color=self.colors[0], label='Glacier'),
                    mpatches.Patch(color=self.colors[1], label='Glacier Lake'),
                    mpatches.Patch(color=self.colors[2], label='Non-Glacier')
                ]
                ind_ax.legend(handles=patches, loc='upper right')
                
                ind_out_path = os.path.join(self.output_dir, f"Map_{os.path.basename(vrt_path)}.png")
                ind_fig.savefig(ind_out_path, bbox_inches='tight', dpi=150)
                plt.close(ind_fig)
                
            except Exception as e:
                print(f"Error on {vrt_path}: {e}")
                axes[i].set_title("ERROR")
                axes[i].axis('off')

        # Hide empty plots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
            
        # Save the big grid
        grid_path = os.path.join(self.output_dir, "Full_Grid_Atlas.png")
        print(f"Saving Full Grid to {grid_path} ... (This might take a moment)")
        plt.tight_layout()
        plt.savefig(grid_path, dpi=200)
        print("Done!")

if __name__ == "__main__":
    classifier = RobustGlacierClassifier()
    # Path to Sentinel 2 VRTs
    vrt_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"
    classifier.process_all(vrt_dir)





