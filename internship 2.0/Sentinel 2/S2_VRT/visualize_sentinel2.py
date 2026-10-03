"""
Quick visualization script for Sentinel-2 VRT files
Using Rasterio (GDAL wrapper)
"""

import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt

vrt_directory = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"

print("="*80)
print("SENTINEL-2 VRT VISUALIZER - Quick View")
print("="*80)

vrt_files = sorted(glob.glob(os.path.join(vrt_directory, "*.vrt")))
print(f"\nFound {len(vrt_files)} VRT files")
print(f"Creating overview visualization...\n")

# Create grid
n_files = len(vrt_files)
cols = 6
rows = (n_files + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(30, rows * 4))
fig.suptitle('Sentinel-2 Satellite Imagery - True Color Composites (2021-2025)', 
            fontsize=20, fontweight='bold')

if n_files == 1:
    axes = np.array([axes])
axes = axes.flatten()

for idx, vrt_file in enumerate(vrt_files):
    try:
        print(f"Process ing {idx+1}/{n_files}: {os.path.basename(vrt_file)}")
        
        with rasterio.open(vrt_file) as src:
            # Sentinel-2 bands: 1=Blue, 2=Green, 3=Red (+ more bands)
            # For true color: Red=Band4, Green=Band3, Blue=Band2
            # But typical stacking is Band3(Red), Band2(Green), Band1(Blue)
            
            # Read RGB bands  
            if src.count >= 3:
                red = src.read(3).astype(float)    # Band 3 = Red
                green = src.read(2).astype(float)  # Band 2 = Green
                blue = src.read(1).astype(float)   # Band 1 = Blue
            else:
                print(f"  Warning: {src.count} bands only")
                continue
            
            # Handle nodata
            nodata = src.nodatavals[0] if src.nodatavals[0] is not None else 0
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
            filename = os.path.basename(vrt_file).replace('Sentinel2_', '').replace('.vrt', '')
            axes[idx].set_title(filename, fontsize=9, fontweight='bold')
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

output_path = os.path.join(vrt_directory, 'sentinel2_overview.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved overview to: {output_path}")

plt.show()

print("\n[SUCCESS] Visualization complete!")





