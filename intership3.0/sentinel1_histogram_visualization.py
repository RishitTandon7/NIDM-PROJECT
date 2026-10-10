"""
Sentinel-1 SAR Histogram Enhanced Visualization
================================================
Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) 
to enhance visibility of features in SAR imagery.

Author: Antigravity
Date: 2026-01-30
"""

import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from skimage import exposure
import math
import warnings

warnings.filterwarnings('ignore')


def load_sentinel1_bands(tiff_path):
    """Load VV, VH, and angle bands from Sentinel-1 GeoTIFF"""
    with rasterio.open(tiff_path) as src:
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)
        angle = src.read(3).astype(np.float32)
        profile = src.profile
    return vv, vh, angle, profile


def convert_to_db(linear_data):
    """Convert linear SAR values to dB scale"""
    eps = 1e-10
    db = 10 * np.log10(np.maximum(linear_data, eps))
    return db


def apply_histogram_equalization(data, clip_limit=0.03):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    for enhanced visualization.
    """
    # Normalize to 0-1 range first
    data_min, data_max = np.nanmin(data), np.nanmax(data)
    if data_max - data_min > 0:
        data_norm = (data - data_min) / (data_max - data_min)
    else:
        data_norm = np.zeros_like(data)
    
    # Apply CLAHE
    data_eq = exposure.equalize_adapthist(data_norm, clip_limit=clip_limit)
    
    return data_eq


def apply_standard_histogram_eq(data):
    """Apply standard histogram equalization"""
    data_min, data_max = np.nanmin(data), np.nanmax(data)
    if data_max - data_min > 0:
        data_norm = (data - data_min) / (data_max - data_min)
    else:
        data_norm = np.zeros_like(data)
    
    data_eq = exposure.equalize_hist(data_norm)
    return data_eq


def create_enhanced_rgb(vv_eq, vh_eq):
    """Create RGB composite from histogram-equalized bands"""
    # VV/VH difference for ratio channel
    ratio = np.abs(vv_eq - vh_eq)
    ratio_eq = apply_histogram_equalization(ratio, clip_limit=0.02)
    
    rgb = np.dstack([
        (vv_eq * 255).astype(np.uint8),
        (vh_eq * 255).astype(np.uint8),
        (ratio_eq * 255).astype(np.uint8)
    ])
    return rgb


def process_histogram_enhanced(tiff_dir, output_dir):
    """
    Process all Sentinel-1 images with histogram equalization
    and create individual + stitched visualizations.
    """
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "S1_*.tif")))
    
    if not tiff_files:
        print("No Sentinel-1 files found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_images = len(tiff_files)
    print(f"Found {num_images} Sentinel-1 images")
    print("=" * 70)
    
    # Storage for mosaic creation
    all_vv_eq = []
    all_vh_eq = []
    all_rgb_eq = []
    all_dates = []
    
    # Process each image
    for i, tiff_path in enumerate(tiff_files):
        filename = os.path.basename(tiff_path)
        date_str = filename.replace('.tif', '').replace('S1_', '')
        all_dates.append(date_str)
        
        print(f"[{i+1}/{num_images}] Processing {date_str}...")
        
        # Load data
        vv, vh, angle, profile = load_sentinel1_bands(tiff_path)
        
        # Convert to dB and clip
        vv_db = np.clip(convert_to_db(vv), -30, 5)
        vh_db = np.clip(convert_to_db(vh), -35, 0)
        
        # Apply CLAHE histogram equalization
        print(f"    Applying histogram equalization...")
        vv_eq = apply_histogram_equalization(vv_db, clip_limit=0.03)
        vh_eq = apply_histogram_equalization(vh_db, clip_limit=0.03)
        
        # Store for mosaic
        all_vv_eq.append(vv_eq)
        all_vh_eq.append(vh_eq)
        
        # Create enhanced RGB
        rgb_eq = create_enhanced_rgb(vv_eq, vh_eq)
        all_rgb_eq.append(rgb_eq)
        
        # Create individual visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Sentinel-1 Histogram Enhanced - {date_str}', fontsize=14, fontweight='bold')
        
        # Row 1: Original (dB scaled)
        ax1 = axes[0, 0]
        ax1.imshow(vv_db, cmap='gray', vmin=-25, vmax=0)
        ax1.set_title('VV Original (dB)')
        ax1.axis('off')
        
        ax2 = axes[0, 1]
        ax2.imshow(vh_db, cmap='gray', vmin=-30, vmax=-5)
        ax2.set_title('VH Original (dB)')
        ax2.axis('off')
        
        ax3 = axes[0, 2]
        # Original RGB
        vv_norm = np.clip((vv_db + 25) / 25, 0, 1)
        vh_norm = np.clip((vh_db + 30) / 25, 0, 1)
        ratio_orig = np.clip((vv_db - vh_db) / 15, 0, 1)
        rgb_orig = np.dstack([vv_norm, vh_norm, ratio_orig])
        ax3.imshow(rgb_orig)
        ax3.set_title('RGB Original')
        ax3.axis('off')
        
        # Row 2: Histogram Equalized
        ax4 = axes[1, 0]
        ax4.imshow(vv_eq, cmap='gray')
        ax4.set_title('VV Histogram Enhanced')
        ax4.axis('off')
        
        ax5 = axes[1, 1]
        ax5.imshow(vh_eq, cmap='gray')
        ax5.set_title('VH Histogram Enhanced')
        ax5.axis('off')
        
        ax6 = axes[1, 2]
        ax6.imshow(rgb_eq)
        ax6.set_title('RGB Histogram Enhanced')
        ax6.axis('off')
        
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, f"S1_HistEq_{date_str}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {out_path}")
    
    # =====================================================
    # CREATE STITCHED MOSAICS
    # =====================================================
    print("\n" + "=" * 70)
    print("Creating Stitched Mosaics...")
    print("=" * 70)
    
    cols = 6
    rows = math.ceil(num_images / cols)
    
    # ----- VV Histogram Enhanced Mosaic -----
    print("\n  Creating VV Histogram Enhanced Mosaic...")
    fig_w, fig_h = cols * 4, rows * 3
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    fig.suptitle('Sentinel-1 VV Backscatter - Histogram Enhanced', fontsize=18, fontweight='bold')
    
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i in range(num_images):
        axes_flat[i].imshow(all_vv_eq[i], cmap='gray')
        axes_flat[i].set_title(all_dates[i], fontsize=9)
        axes_flat[i].axis('off')
    
    # Hide empty subplots
    for i in range(num_images, rows * cols):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    vv_mosaic_path = os.path.join(output_dir, "S1_Mosaic_VV_HistogramEnhanced.png")
    plt.savefig(vv_mosaic_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {vv_mosaic_path}")
    
    # ----- VH Histogram Enhanced Mosaic -----
    print("  Creating VH Histogram Enhanced Mosaic...")
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    fig.suptitle('Sentinel-1 VH Backscatter - Histogram Enhanced', fontsize=18, fontweight='bold')
    
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i in range(num_images):
        axes_flat[i].imshow(all_vh_eq[i], cmap='gray')
        axes_flat[i].set_title(all_dates[i], fontsize=9)
        axes_flat[i].axis('off')
    
    for i in range(num_images, rows * cols):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    vh_mosaic_path = os.path.join(output_dir, "S1_Mosaic_VH_HistogramEnhanced.png")
    plt.savefig(vh_mosaic_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {vh_mosaic_path}")
    
    # ----- RGB Histogram Enhanced Mosaic -----
    print("  Creating RGB Histogram Enhanced Mosaic...")
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    fig.suptitle('Sentinel-1 RGB Composite - Histogram Enhanced', fontsize=18, fontweight='bold')
    
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i in range(num_images):
        axes_flat[i].imshow(all_rgb_eq[i])
        axes_flat[i].set_title(all_dates[i], fontsize=9)
        axes_flat[i].axis('off')
    
    for i in range(num_images, rows * cols):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    rgb_mosaic_path = os.path.join(output_dir, "S1_Mosaic_RGB_HistogramEnhanced.png")
    plt.savefig(rgb_mosaic_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {rgb_mosaic_path}")
    
    # =====================================================
    # CREATE COMPARISON MOSAIC (Original vs Enhanced)
    # =====================================================
    print("\n  Creating Before/After Comparison Grid...")
    
    # Select 6 representative images (every 5th image)
    sample_indices = [0, 5, 10, 15, 20, 25]
    sample_indices = [i for i in sample_indices if i < num_images]
    
    fig, axes = plt.subplots(len(sample_indices), 4, figsize=(20, len(sample_indices) * 4))
    fig.suptitle('Sentinel-1 SAR: Original vs Histogram Enhanced Comparison', fontsize=16, fontweight='bold')
    
    for row, idx in enumerate(sample_indices):
        date_str = all_dates[idx]
        
        # Load original data again for comparison
        tiff_path = tiff_files[idx]
        vv, vh, _, _ = load_sentinel1_bands(tiff_path)
        vv_db = np.clip(convert_to_db(vv), -30, 5)
        vh_db = np.clip(convert_to_db(vh), -35, 0)
        
        # Original VV
        axes[row, 0].imshow(vv_db, cmap='gray', vmin=-25, vmax=0)
        axes[row, 0].set_title(f'{date_str} - VV Original' if row == 0 else 'VV Original', fontsize=10)
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_ylabel(date_str, fontsize=12, rotation=0, labelpad=50)
        
        # Enhanced VV
        axes[row, 1].imshow(all_vv_eq[idx], cmap='gray')
        axes[row, 1].set_title('VV Enhanced' if row == 0 else '', fontsize=10)
        axes[row, 1].axis('off')
        
        # Original VH
        axes[row, 2].imshow(vh_db, cmap='gray', vmin=-30, vmax=-5)
        axes[row, 2].set_title('VH Original' if row == 0 else '', fontsize=10)
        axes[row, 2].axis('off')
        
        # Enhanced VH
        axes[row, 3].imshow(all_vh_eq[idx], cmap='gray')
        axes[row, 3].set_title('VH Enhanced' if row == 0 else '', fontsize=10)
        axes[row, 3].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "S1_Comparison_Original_vs_Enhanced.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {comparison_path}")
    
    print("\n" + "=" * 70)
    print("HISTOGRAM ENHANCEMENT COMPLETE!")
    print(f"Output folder: {output_dir}")
    print("=" * 70)
    
    return vv_mosaic_path, vh_mosaic_path, rgb_mosaic_path


if __name__ == "__main__":
    # Configuration
    TIFF_DIR = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_histogram_enhanced"
    
    process_histogram_enhanced(TIFF_DIR, OUTPUT_DIR)
