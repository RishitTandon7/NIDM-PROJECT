"""
Sentinel-1 SAR Proper Visualization (V3)
=========================================
Data is already in dB - no double conversion!
Proper linear stretch to use full 0-255 grayscale range.

Author: Antigravity
Date: 2026-01-30
"""

import os
import glob
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import uniform_filter
import math
import warnings

warnings.filterwarnings('ignore')


def load_sentinel1_bands(tiff_path):
    """Load VV, VH bands - data is already in dB scale"""
    with rasterio.open(tiff_path) as src:
        vv = src.read(1).astype(np.float32)  # Already in dB
        vh = src.read(2).astype(np.float32)  # Already in dB
    return vv, vh


def lee_filter(img, size=5):
    """Lee speckle filter for noise reduction while preserving edges"""
    img_mean = uniform_filter(img, size)
    img_sqr_mean = uniform_filter(img**2, size)
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = np.var(img)
    img_weights = img_variance / (img_variance + overall_variance + 1e-10)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def proper_linear_stretch(data, low_pct=2, high_pct=98):
    """
    Proper linear stretch using the FULL 0-255 range.
    This preserves all mid-tones and gradients.
    """
    # Get percentile bounds
    valid_data = data[~np.isnan(data)]
    p_low = np.percentile(valid_data, low_pct)
    p_high = np.percentile(valid_data, high_pct)
    
    print(f"      Stretching from [{p_low:.1f}, {p_high:.1f}] dB to [0, 255]")
    
    # Linear stretch to 0-255
    stretched = (data - p_low) / (p_high - p_low)
    stretched = np.clip(stretched, 0, 1)
    
    # Convert to 8-bit
    img_8bit = (stretched * 255).astype(np.uint8)
    
    return img_8bit


def create_sar_image(data, output_path, apply_filter=True):
    """
    Create natural SAR image with proper grayscale range.
    """
    # Optional speckle reduction
    if apply_filter:
        data = lee_filter(data, size=3)
    
    # Proper linear stretch (data already in dB)
    img_8bit = proper_linear_stretch(data, low_pct=1, high_pct=99)
    
    # Save
    img = Image.fromarray(img_8bit, mode='L')
    img.save(output_path, 'PNG')
    
    return img_8bit


def create_stitched_mosaic(images, output_path, cols=6, padding=4):
    """Create stitched mosaic with minimal borders"""
    num_images = len(images)
    rows = math.ceil(num_images / cols)
    
    h, w = images[0].shape
    
    mosaic_h = rows * h + (rows + 1) * padding
    mosaic_w = cols * w + (cols + 1) * padding
    
    # Dark gray border
    mosaic = np.ones((mosaic_h, mosaic_w), dtype=np.uint8) * 30
    
    for i, img in enumerate(images):
        row_idx = i // cols
        col_idx = i % cols
        
        y_start = padding + row_idx * (h + padding)
        x_start = padding + col_idx * (w + padding)
        
        mosaic[y_start:y_start+h, x_start:x_start+w] = img
    
    pil_img = Image.fromarray(mosaic, mode='L')
    pil_img.save(output_path, 'PNG')
    
    print(f"  [OK] Saved: {output_path}")
    return mosaic


def process_proper_visualization(tiff_dir, output_dir):
    """Process all Sentinel-1 images with proper grayscale stretch"""
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "S1_*.tif")))
    
    if not tiff_files:
        print("No Sentinel-1 files found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_images = len(tiff_files)
    print(f"Found {num_images} Sentinel-1 images")
    print("Data is already in dB scale - using proper linear stretch")
    print("=" * 70)
    
    all_vv = []
    all_vh = []
    all_dates = []
    
    for i, tiff_path in enumerate(tiff_files):
        filename = os.path.basename(tiff_path)
        date_str = filename.replace('.tif', '').replace('S1_', '')
        all_dates.append(date_str)
        
        print(f"[{i+1}/{num_images}] Processing {date_str}...")
        
        # Load data (already in dB)
        vv, vh = load_sentinel1_bands(tiff_path)
        
        # Create VV image
        print(f"    VV:")
        vv_out = os.path.join(output_dir, f"S1_VV_{date_str}.png")
        vv_img = create_sar_image(vv, vv_out, apply_filter=True)
        all_vv.append(vv_img)
        
        # Create VH image  
        print(f"    VH:")
        vh_out = os.path.join(output_dir, f"S1_VH_{date_str}.png")
        vh_img = create_sar_image(vh, vh_out, apply_filter=True)
        all_vh.append(vh_img)
    
    # Create mosaics
    print("\n" + "=" * 70)
    print("Creating Stitched Mosaics...")
    print("=" * 70)
    
    vv_mosaic_path = os.path.join(output_dir, "S1_Mosaic_VV.png")
    create_stitched_mosaic(all_vv, vv_mosaic_path, cols=6)
    
    vh_mosaic_path = os.path.join(output_dir, "S1_Mosaic_VH.png")
    create_stitched_mosaic(all_vh, vh_mosaic_path, cols=6)
    
    print("\n" + "=" * 70)
    print("DONE! Full grayscale range now used (0-255)")
    print(f"Output folder: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    TIFF_DIR = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_proper_viz"
    
    process_proper_visualization(TIFF_DIR, OUTPUT_DIR)
