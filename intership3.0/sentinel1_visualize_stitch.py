"""
Sentinel-1 SAR Data Visualization & Stitching
==============================================
Step 1: Visualize and stitch all Sentinel-1 images WITHOUT classification.
Creates a grid mosaic of VV and VH backscatter for all dates.

Author: Antigravity
Date: 2026-01-30
"""

import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import math
import warnings

warnings.filterwarnings('ignore')


def load_sentinel1_bands(tiff_path):
    """Load VV, VH, and angle bands from Sentinel-1 GeoTIFF"""
    with rasterio.open(tiff_path) as src:
        vv = src.read(1).astype(np.float32)  # VV polarization
        vh = src.read(2).astype(np.float32)  # VH polarization
        angle = src.read(3).astype(np.float32)  # Incidence angle
        profile = src.profile
    return vv, vh, angle, profile


def convert_to_db(linear_data):
    """Convert linear SAR values to dB scale"""
    eps = 1e-10
    db = 10 * np.log10(np.maximum(linear_data, eps))
    return db


def normalize_for_display(data, vmin, vmax):
    """Normalize data to 0-255 for display"""
    data_clipped = np.clip(data, vmin, vmax)
    data_norm = (data_clipped - vmin) / (vmax - vmin)
    return (data_norm * 255).astype(np.uint8)


def create_rgb_composite(vv_db, vh_db):
    """
    Create false-color RGB composite from SAR data.
    R: VV, G: VH, B: VV/VH ratio
    """
    vv_norm = normalize_for_display(vv_db, -25, 0)
    vh_norm = normalize_for_display(vh_db, -30, -5)
    ratio = vv_db - vh_db  # VV/VH ratio in dB
    ratio_norm = normalize_for_display(ratio, 0, 15)
    
    rgb = np.dstack([vv_norm, vh_norm, ratio_norm])
    return rgb


def create_individual_images(tiff_dir, output_dir):
    """Create individual PNG for each Sentinel-1 image"""
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "S1_*.tif")))
    
    if not tiff_files:
        print("No Sentinel-1 files found!")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found {len(tiff_files)} Sentinel-1 images")
    print("=" * 60)
    
    image_paths = []
    
    for i, tiff_path in enumerate(tiff_files):
        filename = os.path.basename(tiff_path)
        date_str = filename.replace('.tif', '').replace('S1_', '')
        
        print(f"[{i+1}/{len(tiff_files)}] Processing {date_str}...")
        
        # Load data
        vv, vh, angle, profile = load_sentinel1_bands(tiff_path)
        
        # Convert to dB
        vv_db = np.clip(convert_to_db(vv), -30, 10)
        vh_db = np.clip(convert_to_db(vh), -35, 5)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Sentinel-1 SAR - {date_str}', fontsize=14, fontweight='bold')
        
        # VV
        ax1 = axes[0]
        im1 = ax1.imshow(vv_db, cmap='gray', vmin=-25, vmax=0)
        ax1.set_title('VV Backscatter (dB)')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # VH
        ax2 = axes[1]
        im2 = ax2.imshow(vh_db, cmap='gray', vmin=-30, vmax=-5)
        ax2.set_title('VH Backscatter (dB)')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # RGB Composite
        ax3 = axes[2]
        rgb = create_rgb_composite(vv_db, vh_db)
        ax3.imshow(rgb)
        ax3.set_title('RGB Composite (VV-VH-Ratio)')
        ax3.axis('off')
        
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, f"S1_Raw_{date_str}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        image_paths.append(out_path)
        print(f"    Saved: {out_path}")
    
    return image_paths, tiff_files


def create_stitched_mosaic(tiff_dir, output_dir, cols=5):
    """
    Create a single stitched PNG mosaic of all Sentinel-1 images.
    Shows VV backscatter for each date in a grid.
    """
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "S1_*.tif")))
    
    if not tiff_files:
        print("No Sentinel-1 files found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_images = len(tiff_files)
    rows = math.ceil(num_images / cols)
    
    print(f"\nCreating stitched mosaic: {rows} rows x {cols} cols")
    print("=" * 60)
    
    # First, load one image to get dimensions
    sample_vv, _, _, _ = load_sentinel1_bands(tiff_files[0])
    img_h, img_w = sample_vv.shape
    
    # Create figure
    fig_width = cols * 4
    fig_height = rows * 3
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    fig.suptitle('Sentinel-1 SAR Time Series - VV Backscatter', fontsize=16, fontweight='bold')
    
    # Flatten axes for easy indexing
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    for i, tiff_path in enumerate(tiff_files):
        row_idx = i // cols
        col_idx = i % cols
        
        filename = os.path.basename(tiff_path)
        date_str = filename.replace('.tif', '').replace('S1_', '')
        
        print(f"  Loading {date_str}...")
        
        vv, vh, _, _ = load_sentinel1_bands(tiff_path)
        vv_db = np.clip(convert_to_db(vv), -25, 0)
        
        ax = axes[row_idx][col_idx] if rows > 1 else axes[col_idx]
        ax.imshow(vv_db, cmap='gray', vmin=-25, vmax=0)
        ax.set_title(date_str, fontsize=8)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_images, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        ax = axes[row_idx][col_idx] if rows > 1 else axes[col_idx]
        ax.axis('off')
    
    plt.tight_layout()
    
    mosaic_path = os.path.join(output_dir, "S1_Mosaic_VV_TimeSeries.png")
    plt.savefig(mosaic_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Saved VV Mosaic: {mosaic_path}")
    
    # Create VH mosaic
    print("\nCreating VH mosaic...")
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    fig.suptitle('Sentinel-1 SAR Time Series - VH Backscatter', fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    for i, tiff_path in enumerate(tiff_files):
        row_idx = i // cols
        col_idx = i % cols
        
        filename = os.path.basename(tiff_path)
        date_str = filename.replace('.tif', '').replace('S1_', '')
        
        vv, vh, _, _ = load_sentinel1_bands(tiff_path)
        vh_db = np.clip(convert_to_db(vh), -30, -5)
        
        ax = axes[row_idx][col_idx] if rows > 1 else axes[col_idx]
        ax.imshow(vh_db, cmap='gray', vmin=-30, vmax=-5)
        ax.set_title(date_str, fontsize=8)
        ax.axis('off')
    
    for i in range(num_images, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        ax = axes[row_idx][col_idx] if rows > 1 else axes[col_idx]
        ax.axis('off')
    
    plt.tight_layout()
    
    mosaic_vh_path = os.path.join(output_dir, "S1_Mosaic_VH_TimeSeries.png")
    plt.savefig(mosaic_vh_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved VH Mosaic: {mosaic_vh_path}")
    
    # Create RGB Composite mosaic
    print("\nCreating RGB Composite mosaic...")
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    fig.suptitle('Sentinel-1 SAR Time Series - RGB Composite (VV-VH-Ratio)', fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    for i, tiff_path in enumerate(tiff_files):
        row_idx = i // cols
        col_idx = i % cols
        
        filename = os.path.basename(tiff_path)
        date_str = filename.replace('.tif', '').replace('S1_', '')
        
        vv, vh, _, _ = load_sentinel1_bands(tiff_path)
        vv_db = np.clip(convert_to_db(vv), -25, 0)
        vh_db = np.clip(convert_to_db(vh), -30, -5)
        rgb = create_rgb_composite(vv_db, vh_db)
        
        ax = axes[row_idx][col_idx] if rows > 1 else axes[col_idx]
        ax.imshow(rgb)
        ax.set_title(date_str, fontsize=8)
        ax.axis('off')
    
    for i in range(num_images, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        ax = axes[row_idx][col_idx] if rows > 1 else axes[col_idx]
        ax.axis('off')
    
    plt.tight_layout()
    
    mosaic_rgb_path = os.path.join(output_dir, "S1_Mosaic_RGB_TimeSeries.png")
    plt.savefig(mosaic_rgb_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved RGB Mosaic: {mosaic_rgb_path}")
    
    return mosaic_path, mosaic_vh_path, mosaic_rgb_path


if __name__ == "__main__":
    # Configuration
    TIFF_DIR = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_visualizations"
    
    print("=" * 60)
    print("SENTINEL-1 SAR VISUALIZATION (Step 1 - No Classification)")
    print("=" * 60)
    
    # Create individual images
    print("\n--- Creating Individual Images ---")
    create_individual_images(TIFF_DIR, OUTPUT_DIR)
    
    # Create stitched mosaics
    print("\n--- Creating Stitched Mosaics ---")
    create_stitched_mosaic(TIFF_DIR, OUTPUT_DIR, cols=6)
    
    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE!")
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 60)
