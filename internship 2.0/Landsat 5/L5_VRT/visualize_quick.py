"""
Quick visualization script for Landsat 5 VRT files
Using Rasterio (GDAL wrapper)
"""

import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt

vrt_directory = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Landsat 5\L5_VRT"

print("="*80)
print("LANDSAT 5 VRT VISUALIZER - Quick View")
print("="*80)

vrt_files = sorted(glob.glob(os.path.join(vrt_directory, "*.vrt")))
print(f"\nFound {len(vrt_files)} VRT files")
print(f"Creating overview visualization...\n")

# Create grid
n_files = len(vrt_files)
cols = 5
rows = (n_files + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(25, rows * 4))
fig.suptitle('Landsat 5 Satellite Imagery - Natural Color Composites (1988-2009)', 
            fontsize=18, fontweight='bold')

if n_files == 1:
    axes = np.array([axes])
axes = axes.flatten()

for idx, vrt_file in enumerate(vrt_files):
    try:
        print(f"Processing {idx+1}/{n_files}: {os.path.basename(vrt_file)}")
        
        with rasterio.open(vrt_file) as src:
            # Read RGB bands (3=Red, 2=Green, 1=Blue for natural color)
            red = src.read(3).astype(float)
            green = src.read(2).astype(float)
            blue = src.read(1).astype(float)
            
            # Handle nodata
            nodata = src.nodatavals[0]
            if nodata is not None:
                red[red == nodata] = np.nan
                green[green == nodata] = np.nan
                blue[blue == nodata] = np.nan
            
            # Normalize
            def normalize(band):
                valid_data = band[~np.isnan(band)]
                if len(valid_data) > 0:
                    p2, p98 = np.percentile(valid_data, [2, 98])
                    normalized = (band - p2) / (p98 - p2)
                    return np.clip(normalized, 0, 1)
                return band
            
            rgb = np.dstack((normalize(red), normalize(green), normalize(blue)))
            
            axes[idx].imshow(rgb)
            filename = os.path.basename(vrt_file).replace('Landsat5_', '').replace('.vrt', '')
            axes[idx].set_title(filename, fontsize=10, fontweight='bold')
            axes[idx].axis('off')
            
    except Exception as e:
        print(f"  Error: {str(e)}")
        axes[idx].text(0.5, 0.5, f'Error\n{os.path.basename(vrt_file)}', 
                     ha='center', va='center')
        axes[idx].axis('off')

# Hide unused subplots
for idx in range(n_files, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()

output_path = os.path.join(vrt_directory, 'landsat5_overview.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved overview to: {output_path}")

plt.show()

print("\n[SUCCESS] Visualization complete!")





