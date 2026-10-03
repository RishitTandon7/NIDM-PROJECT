"""
Histogram Analysis for Shadow Removal
1. Calculates Brightness Histogram for a sample Sentinel-2 scene.
2. Identifies the separation between Shadows (dark) and Glaciers (bright).
3. Applies a stricter, histogram-based threshold to remove shadows.
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import glob

def analyze_and_plot_histogram(vrt_path, output_dir):
    s2_date = os.path.basename(vrt_path)[10:20]
    print(f"Analyzing Histogram for {s2_date}...")

    with rasterio.open(vrt_path) as src:
        # Read B2, B3, B4 (RGB)
        try:
            blue  = src.read(2).astype(float)
            green = src.read(3).astype(float)
            red   = src.read(4).astype(float)
        except:
            blue  = src.read(1).astype(float)
            green = src.read(2).astype(float)
            red   = src.read(3).astype(float)

    # 1. Calculate Pixell-Level Brightness
    # (Average of RGB)
    brightness = (blue + green + red) / 3.0
    
    # Flatten for histogram
    # Filter out 0 (No Data)
    valid_pixels = brightness[brightness > 0]
    
    # 2. Plot Histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    counts, bins, patches = ax1.hist(valid_pixels.flatten(), bins=100, color='gray', alpha=0.7, log=True)
    ax1.set_title(f'Brightness Histogram (Log Scale) - {s2_date}')
    ax1.set_xlabel('Pixel Brightness Value')
    ax1.set_ylabel('Frequency (Log)')
    
    # 3. Determine Dynamic Threshold (Simple "Valley" approach or Percentile)
    # Shadows are usually the lower peak, Glaciers are the higher values.
    # Let's visualize a few cutoffs.
    
    # Conservative Shadow Cutoff (e.g. 3500 DN or 0.35 float)
    # If raw DN (0-10000), let's guess shadows are < 2500 (approx)
    
    # Check max value to decide scale
    max_val = np.max(valid_pixels)
    if max_val > 2000: # Raw DN
        shadow_cutoff = 3000
    else: # Float 0-1
        shadow_cutoff = 0.3
        
    ax1.axvline(shadow_cutoff, color='red', linestyle='--', linewidth=2, label=f'Proposed Cutoff ({shadow_cutoff})')
    ax1.legend()

    # 4. Preview the Result Map
    # Filter: Brightness > Cutoff
    glacier_preview = np.zeros_like(brightness)
    glacier_preview[brightness > shadow_cutoff] = 1
    
    ax2.imshow(glacier_preview, cmap='gray')
    ax2.set_title(f'Glacier Mask (Brightness > {shadow_cutoff})')
    ax2.axis('off')

    out_path = os.path.join(output_dir, f"Histogram_Analysis_{s2_date}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved Analysis: {out_path}")

if __name__ == "__main__":
    vrt_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"
    output_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\histogram_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze the 1st image (2021-11-08)
    first_file = sorted(glob.glob(os.path.join(vrt_dir, "*.vrt")))[0]
    analyze_and_plot_histogram(first_file, output_dir)





