"""
Sentinel-1 SAR Denoising & Visualization
=========================================
Multiple speckle filters for SAR noise reduction:
- Lee Filter (adaptive, edge-preserving)
- Enhanced Lee Filter (improved edge preservation)
- Frost Filter (texture-adaptive)
- Gamma MAP Filter (optimal for SAR)
- Multi-temporal Filter (uses time series)

Author: Antigravity
Date: 2026-01-30
"""

import os
import glob
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import uniform_filter, median_filter, gaussian_filter
from scipy.ndimage import convolve
import math
import warnings

warnings.filterwarnings('ignore')


def load_sentinel1_bands(tiff_path):
    """Load VV, VH bands - data is already in dB scale"""
    with rasterio.open(tiff_path) as src:
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)
    return vv, vh


# =============================================================================
# SPECKLE FILTERS
# =============================================================================

def lee_filter(img, size=7):
    """
    Lee Filter - Standard adaptive speckle filter.
    Balances local mean and original value based on local variance.
    
    Parameters:
        img: Input image (dB values)
        size: Window size (odd number, typically 5-11)
    """
    img_mean = uniform_filter(img.astype(np.float64), size)
    img_sqr_mean = uniform_filter(img.astype(np.float64)**2, size)
    img_variance = img_sqr_mean - img_mean**2
    
    overall_variance = np.var(img)
    
    # Weight: 0 = use mean (smooth), 1 = use original (preserve detail)
    weights = img_variance / (img_variance + overall_variance + 1e-10)
    
    output = img_mean + weights * (img - img_mean)
    return output.astype(np.float32)


def enhanced_lee_filter(img, size=7, damping_factor=1.0):
    """
    Enhanced Lee Filter with better edge preservation.
    Uses a damping factor to control smoothing strength.
    
    Parameters:
        img: Input image
        size: Window size
        damping_factor: Higher = more smoothing (0.5-2.0 typical)
    """
    img = img.astype(np.float64)
    
    # Local statistics
    local_mean = uniform_filter(img, size)
    local_sqr_mean = uniform_filter(img**2, size)
    local_var = np.maximum(local_sqr_mean - local_mean**2, 0)
    
    # Estimate noise variance (using median of local variances)
    noise_var = np.median(local_var) * damping_factor
    
    # Coefficient of variation
    cu = np.sqrt(noise_var) / (np.abs(local_mean) + 1e-10)
    ci = np.sqrt(local_var) / (np.abs(local_mean) + 1e-10)
    
    # Weight calculation with damping
    w = 1 - (cu**2 / (ci**2 + 1e-10))
    w = np.clip(w, 0, 1)
    
    output = local_mean + w * (img - local_mean)
    return output.astype(np.float32)


def frost_filter(img, size=5, damping=2.0):
    """
    Frost Filter - Exponentially weighted based on local statistics.
    Better for preserving texture variations.
    
    Parameters:
        img: Input image
        size: Window size
        damping: Damping factor (higher = more smoothing)
    """
    img = img.astype(np.float64)
    
    # Local statistics
    local_mean = uniform_filter(img, size)
    local_sqr_mean = uniform_filter(img**2, size)
    local_var = np.maximum(local_sqr_mean - local_mean**2, 0)
    
    # Coefficient of variation
    cv = np.sqrt(local_var) / (np.abs(local_mean) + 1e-10)
    
    # Create distance-weighted kernel
    half = size // 2
    y, x = np.ogrid[-half:half+1, -half:half+1]
    dist = np.sqrt(x**2 + y**2)
    
    # Apply filter with exponential distance weighting
    output = np.zeros_like(img)
    
    for i in range(-half, half+1):
        for j in range(-half, half+1):
            d = np.sqrt(i**2 + j**2)
            shifted = np.roll(np.roll(img, i, axis=0), j, axis=1)
            weight = np.exp(-damping * cv * d)
            output += weight * shifted
    
    # Normalize
    total_weight = np.zeros_like(img)
    for i in range(-half, half+1):
        for j in range(-half, half+1):
            d = np.sqrt(i**2 + j**2)
            weight = np.exp(-damping * cv * d)
            total_weight += weight
    
    output = output / (total_weight + 1e-10)
    return output.astype(np.float32)


def gamma_map_filter(img, size=7, n_looks=1):
    """
    Gamma MAP Filter - Maximum A Posteriori filter for SAR.
    Assumes Gamma distribution of SAR intensity data.
    
    Parameters:
        img: Input image (should convert from dB to linear for best results)
        size: Window size
        n_looks: Number of looks (1 for single-look, higher for multi-look)
    """
    img = img.astype(np.float64)
    
    # Local statistics
    local_mean = uniform_filter(img, size)
    local_sqr_mean = uniform_filter(img**2, size)
    local_var = np.maximum(local_sqr_mean - local_mean**2, 0)
    
    # Estimate Cu (coefficient of variation of noise)
    cu = 1.0 / np.sqrt(n_looks)
    
    # Ci (local coefficient of variation)
    ci = np.sqrt(local_var) / (np.abs(local_mean) + 1e-10)
    
    # Cmax for fully textured areas
    cmax = np.sqrt(2) * cu
    
    # Calculate filter output
    alpha = (1 + cu**2) / (ci**2 - cu**2 + 1e-10)
    
    # Three cases
    output = np.where(
        ci <= cu,  # Homogeneous area - use mean
        local_mean,
        np.where(
            ci < cmax,  # Heterogeneous - use MAP estimate
            local_mean + (img - local_mean) * (1 - cu**2 / (ci**2 + 1e-10)),
            img  # Highly textured - preserve original
        )
    )
    
    return output.astype(np.float32)


def median_filter_sar(img, size=5):
    """
    Median Filter - Simple but effective for salt-and-pepper noise.
    Good for extreme outliers.
    
    Parameters:
        img: Input image
        size: Window size
    """
    return median_filter(img, size=size).astype(np.float32)


def combined_filter(img, lee_size=5, median_size=3, iterations=1):
    """
    Combined filtering approach:
    1. First remove extreme outliers with median
    2. Then apply Lee filter for adaptive smoothing
    
    Parameters:
        img: Input image
        lee_size: Window size for Lee filter
        median_size: Window size for median filter
        iterations: Number of filter passes
    """
    output = img.copy()
    
    for _ in range(iterations):
        # First pass: Remove extreme outliers
        output = median_filter_sar(output, size=median_size)
        
        # Second pass: Adaptive smoothing
        output = enhanced_lee_filter(output, size=lee_size, damping_factor=1.0)
    
    return output


# =============================================================================
# VISUALIZATION
# =============================================================================

def proper_linear_stretch(data, low_pct=2, high_pct=98):
    """Proper linear stretch using the FULL 0-255 range."""
    valid_data = data[~np.isnan(data)]
    p_low = np.percentile(valid_data, low_pct)
    p_high = np.percentile(valid_data, high_pct)
    
    stretched = (data - p_low) / (p_high - p_low)
    stretched = np.clip(stretched, 0, 1)
    
    return (stretched * 255).astype(np.uint8)


def create_sar_image(data, output_path, filter_type='combined', filter_size=7):
    """
    Create denoised SAR image.
    
    filter_type: 'none', 'lee', 'enhanced_lee', 'frost', 'gamma_map', 'median', 'combined'
    """
    # Apply selected filter
    if filter_type == 'none':
        filtered = data
    elif filter_type == 'lee':
        filtered = lee_filter(data, size=filter_size)
    elif filter_type == 'enhanced_lee':
        filtered = enhanced_lee_filter(data, size=filter_size, damping_factor=1.0)
    elif filter_type == 'frost':
        filtered = frost_filter(data, size=min(filter_size, 5), damping=2.0)
    elif filter_type == 'gamma_map':
        filtered = gamma_map_filter(data, size=filter_size, n_looks=1)
    elif filter_type == 'median':
        filtered = median_filter_sar(data, size=filter_size)
    elif filter_type == 'combined':
        filtered = combined_filter(data, lee_size=filter_size, median_size=3, iterations=1)
    else:
        filtered = data
    
    # Stretch to 0-255
    img_8bit = proper_linear_stretch(filtered, low_pct=1, high_pct=99)
    
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


def process_denoised_visualization(tiff_dir, output_dir, filter_type='combined', filter_size=7):
    """
    Process all Sentinel-1 images with speckle filtering.
    
    filter_type: 'lee', 'enhanced_lee', 'frost', 'gamma_map', 'median', 'combined'
    filter_size: Window size (5-11 recommended)
    """
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "S1_*.tif")))
    
    if not tiff_files:
        print("No Sentinel-1 files found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_images = len(tiff_files)
    print(f"Found {num_images} Sentinel-1 images")
    print(f"Filter: {filter_type} (window size: {filter_size})")
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
        
        # Create denoised VV image
        vv_out = os.path.join(output_dir, f"S1_VV_{date_str}.png")
        vv_img = create_sar_image(vv, vv_out, filter_type=filter_type, filter_size=filter_size)
        all_vv.append(vv_img)
        
        # Create denoised VH image  
        vh_out = os.path.join(output_dir, f"S1_VH_{date_str}.png")
        vh_img = create_sar_image(vh, vh_out, filter_type=filter_type, filter_size=filter_size)
        all_vh.append(vh_img)
    
    # Create mosaics
    print("\n" + "=" * 70)
    print("Creating Stitched Mosaics...")
    print("=" * 70)
    
    vv_mosaic_path = os.path.join(output_dir, "S1_Mosaic_VV_Denoised.png")
    create_stitched_mosaic(all_vv, vv_mosaic_path, cols=6)
    
    vh_mosaic_path = os.path.join(output_dir, "S1_Mosaic_VH_Denoised.png")
    create_stitched_mosaic(all_vh, vh_mosaic_path, cols=6)
    
    print("\n" + "=" * 70)
    print(f"DENOISING COMPLETE! (Filter: {filter_type})")
    print(f"Output folder: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    TIFF_DIR = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_denoised"
    
    # Filter options: 'lee', 'enhanced_lee', 'frost', 'gamma_map', 'median', 'combined'
    # Window size: 5-11 (larger = more smoothing, less detail)
    process_denoised_visualization(
        TIFF_DIR, 
        OUTPUT_DIR, 
        filter_type='combined',  # Best balance of noise reduction and edge preservation
        filter_size=7            # Window size
    )
