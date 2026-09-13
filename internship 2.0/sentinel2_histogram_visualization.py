"""
Sentinel-2 Histogram Visualization and Stitched Mosaic Generator
================================================================
This script performs:
1. Histogram equalization on Sentinel-2 imagery for enhanced visualization
2. Per-band histogram analysis with visualization
3. Creates a final stitched mosaic of all processed images

Author: Automated Processing Pipeline
Date: 2026-01-14
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import glob
from skimage import exposure
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


def normalize_band(band, percentile_low=2, percentile_high=98):
    """
    Normalize a band using percentile-based clipping and rescaling.
    This is more robust than simple min-max normalization.
    """
    band = band.astype(float)
    # Handle no-data values
    valid_mask = band > 0
    if not np.any(valid_mask):
        return np.zeros_like(band)
    
    # Calculate percentiles on valid data only
    low = np.percentile(band[valid_mask], percentile_low)
    high = np.percentile(band[valid_mask], percentile_high)
    
    # Clip and normalize
    band_clipped = np.clip(band, low, high)
    band_normalized = (band_clipped - low) / (high - low + 1e-6)
    
    return band_normalized


def apply_histogram_equalization(image_array, method='adaptive'):
    """
    Apply histogram equalization to enhance image contrast.
    
    Methods:
    - 'standard': Global histogram equalization
    - 'adaptive': CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    if method == 'adaptive':
        # Apply CLAHE for better local contrast
        return exposure.equalize_adapthist(image_array, clip_limit=0.03)
    else:
        return exposure.equalize_hist(image_array)


def create_rgb_composite(tiff_path, apply_equalization=True):
    """
    Create a true-color RGB composite from Sentinel-2 bands.
    Sentinel-2 Band Order: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)...
    """
    with rasterio.open(tiff_path) as src:
        # Read RGB bands (B4=Red, B3=Green, B2=Blue)
        blue = src.read(1).astype(float)   # B2
        green = src.read(2).astype(float)  # B3
        red = src.read(3).astype(float)    # B4
        
    # Normalize each band
    red_norm = normalize_band(red)
    green_norm = normalize_band(green)
    blue_norm = normalize_band(blue)
    
    # Stack to RGB
    rgb = np.dstack([red_norm, green_norm, blue_norm])
    
    if apply_equalization:
        # Apply adaptive histogram equalization to each channel
        for i in range(3):
            rgb[:,:,i] = apply_histogram_equalization(rgb[:,:,i], method='adaptive')
    
    return rgb


def create_false_color_composite(tiff_path, apply_equalization=True):
    """
    Create a false-color composite (NIR, Red, Green) for glacier visibility.
    """
    with rasterio.open(tiff_path) as src:
        green = src.read(2).astype(float)  # B3
        red = src.read(3).astype(float)    # B4
        nir = src.read(4).astype(float)    # B8
        
    # Normalize each band
    nir_norm = normalize_band(nir)
    red_norm = normalize_band(red)
    green_norm = normalize_band(green)
    
    # Stack as false-color (NIR->R, Red->G, Green->B)
    rgb = np.dstack([nir_norm, red_norm, green_norm])
    
    if apply_equalization:
        for i in range(3):
            rgb[:,:,i] = apply_histogram_equalization(rgb[:,:,i], method='adaptive')
    
    return rgb


def plot_band_histograms(tiff_path, output_path):
    """
    Create a detailed histogram visualization for all main bands.
    """
    with rasterio.open(tiff_path) as src:
        blue = src.read(1).astype(float)
        green = src.read(2).astype(float)
        red = src.read(3).astype(float)
        nir = src.read(4).astype(float)
    
    date_str = os.path.basename(tiff_path)[10:20]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Sentinel-2 Band Histogram Analysis - {date_str}', fontsize=16, fontweight='bold')
    
    bands = [blue, green, red, nir]
    band_names = ['Blue (B2)', 'Green (B3)', 'Red (B4)', 'NIR (B8)']
    colors = ['#1E90FF', '#32CD32', '#FF4500', '#8B0000']
    
    for i, (band, name, color) in enumerate(zip(bands, band_names, colors)):
        # Original histogram
        valid_data = band[band > 0].flatten()
        axes[0, i].hist(valid_data, bins=100, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[0, i].set_title(f'{name} - Original', fontweight='bold')
        axes[0, i].set_xlabel('Pixel Value')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].set_yscale('log')
        axes[0, i].axvline(np.mean(valid_data), color='black', linestyle='--', label=f'Mean: {np.mean(valid_data):.0f}')
        axes[0, i].legend(fontsize=8)
        
        # Equalized histogram
        band_norm = normalize_band(band)
        band_eq = apply_histogram_equalization(band_norm)
        valid_eq = band_eq[band > 0].flatten()
        axes[1, i].hist(valid_eq, bins=100, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[1, i].set_title(f'{name} - After Equalization', fontweight='bold')
        axes[1, i].set_xlabel('Pixel Value')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].axvline(np.mean(valid_eq), color='black', linestyle='--', label=f'Mean: {np.mean(valid_eq):.2f}')
        axes[1, i].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def create_comparison_view(tiff_path, output_path):
    """
    Create a side-by-side comparison of original vs histogram-equalized images.
    """
    date_str = os.path.basename(tiff_path)[10:20]
    
    # Create composites
    rgb_original = create_rgb_composite(tiff_path, apply_equalization=False)
    rgb_equalized = create_rgb_composite(tiff_path, apply_equalization=True)
    false_color = create_false_color_composite(tiff_path, apply_equalization=True)
    
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig)
    
    # Original
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(np.clip(rgb_original, 0, 1))
    ax1.set_title(f'Original RGB\n{date_str}', fontweight='bold', fontsize=12)
    ax1.axis('off')
    
    # Equalized
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(np.clip(rgb_equalized, 0, 1))
    ax2.set_title(f'Histogram Equalized RGB\n{date_str}', fontweight='bold', fontsize=12)
    ax2.axis('off')
    
    # False Color
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(np.clip(false_color, 0, 1))
    ax3.set_title(f'False Color (NIR-R-G)\n{date_str}', fontweight='bold', fontsize=12)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def create_stitched_mosaic(tiff_files, output_path, images_per_row=8, thumbnail_size=(300, 300)):
    """
    Create a stitched mosaic of all histogram-equalized Sentinel-2 images.
    """
    print(f"\n{'='*60}")
    print("Creating Stitched Mosaic...")
    print(f"{'='*60}")
    
    n_images = len(tiff_files)
    n_rows = (n_images + images_per_row - 1) // images_per_row
    
    print(f"Total images: {n_images}")
    print(f"Grid: {n_rows} rows x {images_per_row} columns")
    
    # Pre-calculate final mosaic size
    mosaic_width = images_per_row * thumbnail_size[0]
    mosaic_height = n_rows * thumbnail_size[1]
    
    # Create blank mosaic
    mosaic = np.ones((mosaic_height, mosaic_width, 3), dtype=np.uint8) * 30  # Dark gray background
    
    thumbnails = []
    dates = []
    
    for idx, tiff_path in enumerate(tiff_files):
        date_str = os.path.basename(tiff_path)[10:20]
        dates.append(date_str)
        
        print(f"\r  Processing [{idx+1}/{n_images}]: {date_str}...", end="", flush=True)
        
        try:
            # Create histogram-equalized RGB composite
            rgb = create_rgb_composite(tiff_path, apply_equalization=True)
            
            # Convert to PIL for resizing
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(rgb_uint8)
            img_thumb = img.resize(thumbnail_size, Image.LANCZOS)
            thumbnails.append(np.array(img_thumb))
            
        except Exception as e:
            print(f"\n  Error processing {date_str}: {e}")
            # Create placeholder
            placeholder = np.ones((thumbnail_size[1], thumbnail_size[0], 3), dtype=np.uint8) * 128
            thumbnails.append(placeholder)
    
    print("\n  Assembling mosaic...")
    
    # Place thumbnails in mosaic
    for idx, thumb in enumerate(thumbnails):
        row = idx // images_per_row
        col = idx % images_per_row
        
        y_start = row * thumbnail_size[1]
        y_end = y_start + thumbnail_size[1]
        x_start = col * thumbnail_size[0]
        x_end = x_start + thumbnail_size[0]
        
        mosaic[y_start:y_end, x_start:x_end] = thumb
    
    # Save mosaic
    mosaic_img = Image.fromarray(mosaic)
    mosaic_img.save(output_path, quality=95)
    print(f"  Saved mosaic: {output_path}")
    
    return output_path, mosaic


def create_annotated_mosaic(tiff_files, output_path, images_per_row=8):
    """
    Create a final annotated mosaic with date labels using matplotlib.
    This version includes proper annotations and a professional look.
    """
    print(f"\n{'='*60}")
    print("Creating Annotated Mosaic with Labels...")
    print(f"{'='*60}")
    
    n_images = len(tiff_files)
    n_rows = (n_images + images_per_row - 1) // images_per_row
    
    fig, axes = plt.subplots(n_rows, images_per_row, figsize=(images_per_row * 3, n_rows * 3))
    fig.suptitle('Sentinel-2 Histogram Equalized Time Series\nGlacier Region Analysis', 
                 fontsize=20, fontweight='bold', y=1.02)
    
    # Flatten axes for easy iteration
    if n_rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()
    
    for idx, tiff_path in enumerate(tiff_files):
        date_str = os.path.basename(tiff_path)[10:20]
        print(f"\r  Processing [{idx+1}/{n_images}]: {date_str}...", end="", flush=True)
        
        try:
            rgb = create_rgb_composite(tiff_path, apply_equalization=True)
            axes[idx].imshow(np.clip(rgb, 0, 1))
            axes[idx].set_title(date_str, fontsize=9, fontweight='bold', pad=2)
            axes[idx].axis('off')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error:\n{date_str}', 
                          ha='center', va='center', fontsize=8)
            axes[idx].axis('off')
    
    # Hide unused axes
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n  Saved annotated mosaic: {output_path}")
    return output_path


def create_histogram_summary_grid(tiff_files, output_path, sample_count=9):
    """
    Create a summary grid showing histogram analysis for sample images.
    """
    print(f"\n{'='*60}")
    print("Creating Histogram Summary Grid...")
    print(f"{'='*60}")
    
    # Select evenly spaced samples
    n_images = len(tiff_files)
    sample_indices = np.linspace(0, n_images - 1, sample_count).astype(int)
    sample_files = [tiff_files[i] for i in sample_indices]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle('Sentinel-2 Histogram Visualization - Sample Analysis\n(Histogram Equalized)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    for idx, (ax, tiff_path) in enumerate(zip(axes.flatten(), sample_files)):
        date_str = os.path.basename(tiff_path)[10:20]
        print(f"\r  Processing sample [{idx+1}/{sample_count}]: {date_str}...", end="", flush=True)
        
        try:
            # Create RGB composite
            rgb = create_rgb_composite(tiff_path, apply_equalization=True)
            
            # Create inset histogram
            ax.imshow(np.clip(rgb, 0, 1))
            ax.set_title(date_str, fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Add small histogram inset
            inset_ax = ax.inset_axes([0.02, 0.02, 0.35, 0.25])
            
            # Get brightness histogram
            brightness = (rgb[:,:,0] + rgb[:,:,1] + rgb[:,:,2]) / 3.0
            valid = brightness[brightness > 0].flatten()
            inset_ax.hist(valid, bins=50, color='cyan', alpha=0.8, edgecolor='none')
            inset_ax.set_facecolor('#1a1a2e')
            inset_ax.tick_params(colors='white', labelsize=6)
            inset_ax.spines['bottom'].set_color('white')
            inset_ax.spines['left'].set_color('white')
            inset_ax.spines['top'].set_visible(False)
            inset_ax.spines['right'].set_visible(False)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error', ha='center', va='center')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0f0f1a', edgecolor='none')
    plt.close()
    
    print(f"\n  Saved histogram summary: {output_path}")
    return output_path


def main():
    """
    Main execution function.
    """
    # Configuration
    tiff_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    output_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\segmented_outputs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all Sentinel-2 TIFF files
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "Sentinel2_*.tif")))
    
    if not tiff_files:
        print("ERROR: No Sentinel-2 TIFF files found!")
        return
    
    print(f"""
{'='*70}
   SENTINEL-2 HISTOGRAM VISUALIZATION AND MOSAIC GENERATOR (SEGMENTED)
{'='*70}
   Input Directory:  {tiff_dir}
   Output Directory: {output_dir}
   Total Images:     {len(tiff_files)}
{'='*70}
""")
    
    # 1. Create detailed band histograms for first image (sample)
    print("\n[STEP 1/3] Creating Band Histogram Analysis (Sample)...")
    sample_tiff = tiff_files[0]
    histogram_output = os.path.join(output_dir, "band_histogram_analysis.png")
    plot_band_histograms(sample_tiff, histogram_output)
    print(f"  [OK] Saved: {histogram_output}")
    
    # 2. Create comparison view for sample
    print("\n[STEP 2/3] Creating Before/After Comparison (Sample)...")
    comparison_output = os.path.join(output_dir, "histogram_equalization_comparison.png")
    create_comparison_view(sample_tiff, comparison_output)
    print(f"  [OK] Saved: {comparison_output}")
    
    # 3. Create Segmented Mosaics
    print("\n[STEP 3/3] Creating Segmented Mosaics...")
    
    # Define Segments as requested:
    # 20 images (4 cols & 5 rows) * 3 segments
    # 6 images * 1 segment
    segments = [
        {"name": "Segment_1", "start": 0, "count": 20, "cols": 4},
        {"name": "Segment_2", "start": 20, "count": 20, "cols": 4},
        {"name": "Segment_3", "start": 40, "count": 20, "cols": 4},
        {"name": "Segment_4", "start": 60, "count": 6, "cols": 3},
    ]

    for seg in segments:
        start = seg["start"]
        end = start + seg["count"]
        batch_files = tiff_files[start:end]
        
        if not batch_files:
            continue
            
        print(f"\n  Processing Mosaic for {seg['name']} ({len(batch_files)} images)...")
        
        # Annotated Mosaic
        mosaic_output = os.path.join(output_dir, f"Mosaic_Histogram_{seg['name']}.png")
        create_annotated_mosaic(batch_files, mosaic_output, images_per_row=seg["cols"])
        
        # Simple Mosaic (optional, but good for high res)
        # simple_output = os.path.join(output_dir, f"Simple_Mosaic_{seg['name']}.png")
        # create_stitched_mosaic(batch_files, simple_output, images_per_row=seg["cols"], thumbnail_size=(400,400))
        
    print(f"""
{'='*70}
   PROCESSING COMPLETE
{'='*70}
   Output files saved in: {output_dir}
{'='*70}
""")


if __name__ == "__main__":
    main()



