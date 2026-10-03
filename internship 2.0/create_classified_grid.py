"""
Create classified glacier maps grid for all 66 Sentinel-2 scenes
Shows Glacier (Gray), Glacier Lake (Red), Non-Glacier (Blue)
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob

def classify_scene(vrt_path):
    """Quick classification of a Sentinel-2 scene"""
    
    with rasterio.open(vrt_path) as src:
        # Read bands
        bands = []
        for i in range(1, min(src.count + 1, 11)):
            band = src.read(i).astype(float)
            nodata = src.nodatavals[i-1] if src.nodatavals[i-1] is not None else 0
            band[band == nodata] = 0
            band[band < 0] = 0
            bands.append(band)
        
        height, width = bands[0].shape
        
        # Calculate indices
        green = bands[2]  # Band 3
        nir = bands[7] if len(bands) > 7 else bands[4]
        swir = bands[9] if len(bands) > 9 else bands[5]
        red = bands[3] if len(bands) > 3 else bands[2]
        
        # NDWI - water detection
        ndwi = np.divide(green - nir, green + nir + 1e-10, 
                       out=np.zeros_like(green), where=(green+nir)!=0)
        
        # NDSI - ice/snow detection
        ndsi = np.divide(green - swir, green + swir + 1e-10,
                       out=np.zeros_like(green), where=(green+swir)!=0)
        
        # NDVI - vegetation
        ndvi = np.divide(nir - red, nir + red + 1e-10,
                       out=np.zeros_like(nir), where=(nir+red)!=0)
        
        # Brightness
        brightness = np.mean(bands[:min(len(bands), 4)], axis=0)
        
        # CLASSIFICATION
        classified = np.zeros((height, width), dtype=np.uint8)
        
        # Class 0: GLACIER (gray)
        glacier_mask = (ndsi > 0.4) & (brightness > np.percentile(brightness, 50)) & (ndvi < 0.2)
        classified[glacier_mask] = 0
        
        # Class 1: GLACIER LAKE (red/pink)
        glacier_lake_mask = (
            (ndwi > 0.2) & 
            (ndsi > -0.1) & (ndsi < 0.5) &
            (brightness < np.percentile(brightness, 60)) &
            (ndvi < 0.3)
        )
        classified[glacier_lake_mask] = 1
        
        # Class 2: NON-GLACIER (blue)
        non_glacier_mask = ~(glacier_mask | glacier_lake_mask)
        classified[non_glacier_mask] = 2
        
        # Refinements
        dark_water = (ndwi > 0.3) & (brightness < np.percentile(brightness, 30))
        classified[dark_water] = 1
        
        very_bright = brightness > np.percentile(brightness, 85)
        classified[very_bright & (ndsi > 0.2)] = 0
        
        return classified


def create_classified_grid():
    """Create grid of all classified scenes"""
    
    print("="*80)
    print("CLASSIFIED GLACIER MAPS GRID - ALL 66 SENTINEL-2 SCENES")
    print("="*80)
    
    # Load VRT files
    vrt_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"
    vrt_files = sorted(glob.glob(os.path.join(vrt_dir, "*.vrt")))
    
    print(f"\nFound {len(vrt_files)} scenes to classify")
    print("Processing and classifying all scenes...")
    
    n_images = len(vrt_files)
    cols = 6
    rows = (n_images + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
    fig.suptitle('Classified Glacier Maps - Sentinel-2 Time Series (2021-2025)\nGlacier (Gray) | Glacier Lake (Red) | Non-Glacier (Blue)', 
                fontsize=18, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    # Color map
    cmap = ListedColormap(['#808080', '#FF6B6B', '#4A90E2'])  # Gray, Red, Blue
    
    # Process each scene
    for idx, vrt_path in enumerate(vrt_files):
        print(f"  [{idx+1}/{n_images}] Classifying: {os.path.basename(vrt_path)}")
        
        try:
            # Classify
            classified = classify_scene(vrt_path)
            
            # Display
            axes[idx].imshow(classified, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
            
            # Extract date
            filename = os.path.basename(vrt_path)
            date_str = filename.replace('Sentinel2_', '').replace('.vrt', '')
            
            axes[idx].set_title(date_str, fontsize=9, fontweight='bold')
            axes[idx].axis('off')
            
            # Print stats for first image
            if idx == 0:
                total = classified.size
                print(f"\n    Sample classification (first scene):")
                print(f"      Glacier: {np.sum(classified==0)/total*100:.1f}%")
                print(f"      Glacier Lake: {np.sum(classified==1)/total*100:.1f}%")
                print(f"      Non-Glacier: {np.sum(classified==2)/total*100:.1f}%\n")
            
        except Exception as e:
            print(f"    Error: {e}")
            axes[idx].text(0.5, 0.5, f'Error\n{os.path.basename(vrt_path)}',
                         ha='center', va='center', fontsize=8)
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', label='Glacier', edgecolor='black'),
        Patch(facecolor='#FF6B6B', label='Glacier Lake', edgecolor='black'),
        Patch(facecolor='#4A90E2', label='Non-Glacier', edgecolor='black')
    ]
    fig.legend(handles=legend_elements, loc='upper right', 
              bbox_to_anchor=(0.98, 0.98), fontsize=14, framealpha=0.9)
    
    # Layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    output_dir = "classified_glacier_maps"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'classified_66_scenes_grid.png')
    
    print(f"\nSaving classified grid...")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    
    plt.show()
    plt.close()
    
    # Also save individual maps
    print(f"\nGenerating individual classified maps...")
    individual_dir = os.path.join(output_dir, "individual_maps")
    os.makedirs(individual_dir, exist_ok=True)
    
    for idx, vrt_path in enumerate(vrt_files):
        try:
            classified = classify_scene(vrt_path)
            filename = os.path.basename(vrt_path).replace('.vrt', '_classified.png')
            
            # Create individual map
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(classified, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
            
            date_str = filename.replace('Sentinel2_', '').replace('_classified.png', '')
            ax.set_title(f'Classified Glacier Map - {date_str}', 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Legend
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            individual_path = os.path.join(individual_dir, filename)
            plt.savefig(individual_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            if (idx + 1) % 10 == 0:
                print(f"  Saved {idx+1}/{n_images} individual maps")
                
        except Exception as e:
            print(f"  Error saving {filename}: {e}")
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"  Grid saved: {output_path}")
    print(f"  Individual maps: {individual_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    create_classified_grid()





