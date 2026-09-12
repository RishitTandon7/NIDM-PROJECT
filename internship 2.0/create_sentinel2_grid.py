"""
Create a stitched grid overview of all 66 Sentinel-2 images
Shows temporal progression of glacier area
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import glob
from datetime import datetime

def create_sentinel2_grid():
    """Create grid of all Sentinel-2 scenes"""
    
    print("="*80)
    print("SENTINEL-2 TIME SERIES GRID GENERATOR")
    print("="*80)
    
    # Load all VRT files
    vrt_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"
    vrt_files = sorted(glob.glob(os.path.join(vrt_dir, "*.vrt")))
    
    print(f"\nFound {len(vrt_files)} Sentinel-2 scenes")
    print(f"Date range: {os.path.basename(vrt_files[0])} to {os.path.basename(vrt_files[-1])}")
    
    n_images = len(vrt_files)
    
    # Grid layout
    cols = 6
    rows = (n_images + cols - 1) // cols
    
    print(f"\nCreating {rows}x{cols} grid...")
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
    fig.suptitle('Sentinel-2 Glacier Monitoring Time Series\n66 Scenes (2021-2025)', 
                fontsize=18, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    # Process each scene
    for idx, vrt_path in enumerate(vrt_files):
        print(f"Processing {idx+1}/{n_images}: {os.path.basename(vrt_path)}")
        
        try:
            with rasterio.open(vrt_path) as src:
                # Read RGB bands for natural color
                # Sentinel-2: Band 4 (Red), Band 3 (Green), Band 2 (Blue)
                red = src.read(4).astype(float) if src.count >= 4 else src.read(3).astype(float)
                green = src.read(3).astype(float) if src.count >= 3 else src.read(2).astype(float)
                blue = src.read(2).astype(float) if src.count >= 2 else src.read(1).astype(float)
                
                # Handle nodata
                for band in [red, green, blue]:
                    band[band < 0] = 0
                    band[band > 10000] = 10000
                
                # Normalize (percentile stretch for better visualization)
                def normalize(band):
                    valid = band[band > 0]
                    if len(valid) > 0:
                        p2, p98 = np.percentile(valid, [2, 98])
                        normalized = (band - p2) / (p98 - p2 + 1e-10)
                        return np.clip(normalized, 0, 1)
                    return band / (band.max() + 1e-10)
                
                rgb = np.dstack((normalize(red), normalize(green), normalize(blue)))
                
                # Display
                axes[idx].imshow(rgb)
                
                # Extract date from filename
                filename = os.path.basename(vrt_path)
                date_str = filename.replace('Sentinel2_', '').replace('.vrt', '')
                
                axes[idx].set_title(date_str, fontsize=9, fontweight='bold')
                axes[idx].axis('off')
                
        except Exception as e:
            print(f"  Error: {e}")
            axes[idx].text(0.5, 0.5, f'Error\n{os.path.basename(vrt_path)}',
                         ha='center', va='center', fontsize=8)
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    # Layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    output_dir = "sentinel2_time_series"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'sentinel2_66_scenes_grid.png')
    
    print(f"\nSaving high-resolution grid...")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    plt.show()
    plt.close()
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    create_sentinel2_grid()



