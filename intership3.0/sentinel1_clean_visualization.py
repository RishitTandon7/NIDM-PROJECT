"""
Sentinel-1 SAR Clean Visualization
===================================
Creates clean, high-contrast SAR images with visible texture
- No decorations, colorbars, or titles
- Full dynamic range with percentile stretching
- Enhanced speckle texture visibility
- Professional SAR imagery output

Author: Antigravity
Date: 2026-01-30
"""

import os
import glob
import numpy as np
import rasterio
from PIL import Image
from skimage import exposure
import math
import warnings

warnings.filterwarnings('ignore')


def load_sentinel1_bands(tiff_path):
    """Load VV, VH bands from Sentinel-1 GeoTIFF"""
    with rasterio.open(tiff_path) as src:
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)
    return vv, vh


def convert_to_db(linear_data):
    """Convert linear SAR values to dB scale"""
    eps = 1e-10
    db = 10 * np.log10(np.maximum(linear_data, eps))
    return db


def percentile_stretch(data, low_pct=2, high_pct=98):
    """
    Apply percentile-based contrast stretching.
    This gives better dynamic range than fixed min/max.
    """
    p_low = np.percentile(data[~np.isnan(data)], low_pct)
    p_high = np.percentile(data[~np.isnan(data)], high_pct)
    
    stretched = (data - p_low) / (p_high - p_low)
    stretched = np.clip(stretched, 0, 1)
    
    return stretched


def apply_aggressive_enhancement(data, clip_limit=0.02):
    """
    Apply aggressive CLAHE for high-contrast output while preserving texture.
    """
    # First normalize
    data_norm = percentile_stretch(data, low_pct=1, high_pct=99)
    
    # Apply CLAHE with moderate clip limit to preserve speckle texture
    enhanced = exposure.equalize_adapthist(data_norm, 
                                            kernel_size=None,  # Auto
                                            clip_limit=clip_limit,
                                            nbins=256)
    return enhanced


def create_clean_sar_image(data, output_path, enhance_mode='clahe'):
    """
    Create a clean grayscale SAR image without any decorations.
    
    enhance_mode:
        - 'percentile': Simple percentile stretching
        - 'clahe': CLAHE enhancement
        - 'combined': Percentile + CLAHE
    """
    if enhance_mode == 'percentile':
        enhanced = percentile_stretch(data, low_pct=1, high_pct=99)
    elif enhance_mode == 'clahe':
        enhanced = apply_aggressive_enhancement(data, clip_limit=0.015)
    elif enhance_mode == 'combined':
        stretched = percentile_stretch(data, low_pct=2, high_pct=98)
        enhanced = exposure.equalize_adapthist(stretched, clip_limit=0.02)
    else:
        enhanced = percentile_stretch(data)
    
    # Convert to 8-bit
    img_8bit = (enhanced * 255).astype(np.uint8)
    
    # Save using PIL for clean output
    img = Image.fromarray(img_8bit, mode='L')
    img.save(output_path, 'PNG')
    
    return img_8bit


def create_stitched_mosaic_clean(images, dates, output_path, cols=6):
    """
    Create a stitched mosaic of clean SAR images with minimal borders.
    """
    num_images = len(images)
    rows = math.ceil(num_images / cols)
    
    # Get dimensions from first image
    h, w = images[0].shape
    
    # Add small padding/border
    padding = 2
    
    # Calculate mosaic dimensions
    mosaic_h = rows * h + (rows + 1) * padding
    mosaic_w = cols * w + (cols + 1) * padding
    
    # Create white background mosaic
    mosaic = np.ones((mosaic_h, mosaic_w), dtype=np.uint8) * 30  # Dark gray background
    
    for i, img in enumerate(images):
        row_idx = i // cols
        col_idx = i % cols
        
        y_start = padding + row_idx * (h + padding)
        x_start = padding + col_idx * (w + padding)
        
        mosaic[y_start:y_start+h, x_start:x_start+w] = img
    
    # Save mosaic
    pil_img = Image.fromarray(mosaic, mode='L')
    pil_img.save(output_path, 'PNG')
    
    print(f"  [OK] Saved mosaic: {output_path}")
    return mosaic


def process_clean_visualization(tiff_dir, output_dir):
    """
    Process all Sentinel-1 images with clean, professional visualization.
    """
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "S1_*.tif")))
    
    if not tiff_files:
        print("No Sentinel-1 files found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_images = len(tiff_files)
    print(f"Found {num_images} Sentinel-1 images")
    print("=" * 70)
    
    # Storage for mosaics
    all_vv = []
    all_vh = []
    all_dates = []
    
    # Process each image
    for i, tiff_path in enumerate(tiff_files):
        filename = os.path.basename(tiff_path)
        date_str = filename.replace('.tif', '').replace('S1_', '')
        all_dates.append(date_str)
        
        print(f"[{i+1}/{num_images}] Processing {date_str}...")
        
        # Load data
        vv, vh = load_sentinel1_bands(tiff_path)
        
        # Convert to dB
        vv_db = convert_to_db(vv)
        vh_db = convert_to_db(vh)
        
        # Create clean VV image
        vv_out = os.path.join(output_dir, f"S1_VV_{date_str}.png")
        vv_enhanced = create_clean_sar_image(vv_db, vv_out, enhance_mode='combined')
        all_vv.append(vv_enhanced)
        print(f"    VV saved: {vv_out}")
        
        # Create clean VH image
        vh_out = os.path.join(output_dir, f"S1_VH_{date_str}.png")
        vh_enhanced = create_clean_sar_image(vh_db, vh_out, enhance_mode='combined')
        all_vh.append(vh_enhanced)
        print(f"    VH saved: {vh_out}")
    
    # =====================================================
    # CREATE STITCHED MOSAICS
    # =====================================================
    print("\n" + "=" * 70)
    print("Creating Stitched Mosaics...")
    print("=" * 70)
    
    # VV Mosaic
    print("\n  Creating VV Mosaic...")
    vv_mosaic_path = os.path.join(output_dir, "S1_Mosaic_VV_Clean.png")
    create_stitched_mosaic_clean(all_vv, all_dates, vv_mosaic_path, cols=6)
    
    # VH Mosaic
    print("  Creating VH Mosaic...")
    vh_mosaic_path = os.path.join(output_dir, "S1_Mosaic_VH_Clean.png")
    create_stitched_mosaic_clean(all_vh, all_dates, vh_mosaic_path, cols=6)
    
    print("\n" + "=" * 70)
    print("CLEAN SAR VISUALIZATION COMPLETE!")
    print(f"Output folder: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    # Configuration
    TIFF_DIR = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_clean_visualization"
    
    process_clean_visualization(TIFF_DIR, OUTPUT_DIR)
