"""
Sentinel-1 SAR Natural Visualization (V2)
==========================================
Creates natural-looking SAR images matching reference style:
- Full grayscale range with visible texture
- Natural speckle pattern preserved
- Proper mid-tone distribution
- Clean output without decorations

Author: Antigravity
Date: 2026-01-30
"""

import os
import glob
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import uniform_filter
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


def lee_filter(img, size=7):
    """
    Apply Lee speckle filter to reduce noise while preserving edges.
    This is a standard SAR denoising filter.
    """
    img_mean = uniform_filter(img, size)
    img_sqr_mean = uniform_filter(img**2, size)
    img_variance = img_sqr_mean - img_mean**2
    
    overall_variance = np.var(img)
    
    img_weights = img_variance / (img_variance + overall_variance + 1e-10)
    img_output = img_mean + img_weights * (img - img_mean)
    
    return img_output


def natural_sar_enhancement(data, low_pct=0.5, high_pct=99.5, gamma=0.8):
    """
    Create natural-looking SAR imagery with:
    - Proper percentile stretching
    - Gamma correction for mid-tone enhancement
    - Full dynamic range utilization
    """
    # Handle NaN and Inf
    data = np.nan_to_num(data, nan=np.nanmedian(data), posinf=0, neginf=-50)
    
    # Percentile-based clipping
    p_low = np.percentile(data, low_pct)
    p_high = np.percentile(data, high_pct)
    
    # Normalize to 0-1
    normalized = (data - p_low) / (p_high - p_low + 1e-10)
    normalized = np.clip(normalized, 0, 1)
    
    # Apply gamma correction to enhance mid-tones
    # gamma < 1 brightens mid-tones
    enhanced = np.power(normalized, gamma)
    
    return enhanced


def natural_sar_with_texture(data, speckle_reduce=True):
    """
    Create SAR visualization that looks like the reference images:
    - Visible texture/speckle
    - Good contrast between features
    - Natural gray gradients
    """
    # Convert to dB if not already
    data_db = data.copy()
    
    # Optional: Light speckle reduction (preserves texture)
    if speckle_reduce:
        data_db = lee_filter(data_db, size=3)  # Small window to preserve texture
    
    # Natural enhancement with proper gamma
    enhanced = natural_sar_enhancement(data_db, 
                                        low_pct=1, 
                                        high_pct=99, 
                                        gamma=0.7)  # Gamma < 1 brightens mid-tones
    
    return enhanced


def create_natural_sar_image(data, output_path, apply_filter=True):
    """
    Create publication-quality SAR grayscale image.
    """
    # Enhanced visualization
    enhanced = natural_sar_with_texture(data, speckle_reduce=apply_filter)
    
    # Convert to 8-bit
    img_8bit = (enhanced * 255).astype(np.uint8)
    
    # Save using PIL
    img = Image.fromarray(img_8bit, mode='L')
    img.save(output_path, 'PNG')
    
    return img_8bit


def create_stitched_mosaic(images, output_path, cols=6, padding=4):
    """
    Create a clean stitched mosaic with minimal borders.
    """
    num_images = len(images)
    rows = math.ceil(num_images / cols)
    
    h, w = images[0].shape
    
    mosaic_h = rows * h + (rows + 1) * padding
    mosaic_w = cols * w + (cols + 1) * padding
    
    # Dark gray border/background
    mosaic = np.ones((mosaic_h, mosaic_w), dtype=np.uint8) * 40
    
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


def process_natural_visualization(tiff_dir, output_dir):
    """
    Process all Sentinel-1 images with natural SAR visualization.
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
        
        # Create natural VV image
        vv_out = os.path.join(output_dir, f"S1_VV_{date_str}.png")
        vv_natural = create_natural_sar_image(vv_db, vv_out, apply_filter=True)
        all_vv.append(vv_natural)
        
        # Create natural VH image
        vh_out = os.path.join(output_dir, f"S1_VH_{date_str}.png")
        vh_natural = create_natural_sar_image(vh_db, vh_out, apply_filter=True)
        all_vh.append(vh_natural)
        
        print(f"    Saved: VV, VH")
    
    # Create stitched mosaics
    print("\n" + "=" * 70)
    print("Creating Stitched Mosaics...")
    print("=" * 70)
    
    # VV Mosaic
    print("\n  Creating VV Time Series Mosaic...")
    vv_mosaic_path = os.path.join(output_dir, "S1_Mosaic_VV_Natural.png")
    create_stitched_mosaic(all_vv, vv_mosaic_path, cols=6)
    
    # VH Mosaic
    print("  Creating VH Time Series Mosaic...")
    vh_mosaic_path = os.path.join(output_dir, "S1_Mosaic_VH_Natural.png")
    create_stitched_mosaic(all_vh, vh_mosaic_path, cols=6)
    
    print("\n" + "=" * 70)
    print("NATURAL SAR VISUALIZATION COMPLETE!")
    print(f"Output folder: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    TIFF_DIR = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_natural_viz"
    
    process_natural_visualization(TIFF_DIR, OUTPUT_DIR)
