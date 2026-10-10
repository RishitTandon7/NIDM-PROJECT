"""
YOLO-Enhanced Glacial Lake Detection - SINGLE IMAGE TEST
=========================================================
This script tests the 4-step pipeline on ONE image:
1. YOLOv8 → Detect potential lake regions
2. NDWI/MNDWI → Verify water bodies
3. NDSI + DEM → Mask out glaciers
4. Classifier → Distinguish Glacial vs Non-glacial lakes

Author: Antigravity
Date: 2026-01-23
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import cv2
from ultralytics import YOLO
import torch
import geopandas as gpd
from skimage import exposure
from scipy.ndimage import label, binary_opening
import warnings

warnings.filterwarnings('ignore')


class YOLOGlacialLakeDetector:
    def __init__(self, model_size='n'):
        """
        Initialize YOLO detector
        model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        """
        print(f"Initializing YOLOv8-{model_size} detector...")
        self.model = YOLO(f'yolov8{model_size}-seg.pt')  # Segmentation model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
    
    def load_and_enhance_bands(self, tiff_path):
        """Load and enhance Sentinel-2 bands"""
        with rasterio.open(tiff_path) as src:
            profile = src.profile
            
            # Load bands
            blue = src.read(1).astype(np.float32)
            green = src.read(2).astype(np.float32)
            red = src.read(3).astype(np.float32)
            nir = src.read(4).astype(np.float32)
            
            # SWIR
            if src.count >= 9:
                swir = src.read(9).astype(np.float32)
            elif src.count >= 5:
                swir = src.read(5).astype(np.float32)
            else:
                swir = np.zeros_like(blue)
        
        return {
            'blue': blue,
            'green': green,
            'red': red,
            'nir': nir,
            'swir': swir,
            'profile': profile
        }
    
    def create_rgb_for_yolo(self, bands):
        """Create RGB image for YOLO detection"""
        # Normalize using percentiles
        def normalize(band):
            valid = band[band > 0]
            if len(valid) == 0:
                return np.zeros_like(band, dtype=np.uint8)
            p2, p98 = np.percentile(valid, [2, 98])
            band_norm = np.clip((band - p2) / (p98 - p2 + 1e-10), 0, 1)
            # Apply CLAHE for better contrast
            band_norm = exposure.equalize_adapthist(band_norm, clip_limit=0.03)
            return (band_norm * 255).astype(np.uint8)
        
        r = normalize(bands['red'])
        g = normalize(bands['green'])
        b = normalize(bands['blue'])
        
        rgb = np.dstack([r, g, b])
        return rgb
    
    def detect_with_yolo(self, rgb_image):
        """
        Step 1: Use YOLO to detect potential water bodies
        We'll use the 'water' class or general object detection
        """
        print("\n[STEP 1] YOLO Detection...")
        
        # Run YOLO detection
        results = self.model.predict(
            rgb_image,
            conf=0.25,  # Confidence threshold
            iou=0.45,   # NMS IoU threshold
            device=self.device,
            verbose=False
        )
        
        # Create detection mask
        h, w = rgb_image.shape[:2]
        detection_mask = np.zeros((h, w), dtype=np.uint8)
        
        detections = []
        
        if len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.data.cpu().numpy()
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                # Resize mask to image size
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                detection_mask = np.maximum(detection_mask, mask_binary)
                
                x1, y1, x2, y2, conf, cls = box
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': int(cls)
                })
        
        print(f"  Found {len(detections)} potential regions")
        return detection_mask, detections
    
    def verify_water_with_indices(self, bands, detection_mask):
        """
        Step 2: Verify detected regions are actually water using NDWI/MNDWI
        """
        print("\n[STEP 2] Water Verification (NDWI/MNDWI)...")
        
        green = bands['green']
        nir = bands['nir']
        swir = bands['swir']
        
        eps = 1e-10
        
        # Calculate water indices
        ndwi = (green - nir) / (green + nir + eps)
        mndwi = (green - swir) / (green + swir + eps)
        
        # Water verification: High NDWI or MNDWI
        water_mask = (
            ((ndwi > 0.15) | (mndwi > 0.1)) &
            (detection_mask > 0)  # Only in detected regions
        )
        
        # Clean up
        water_mask = binary_opening(water_mask, structure=np.ones((3, 3)))
        
        water_pct = (np.sum(water_mask) / np.sum(detection_mask > 0) * 100) if np.sum(detection_mask > 0) > 0 else 0
        print(f"  Water verification: {water_pct:.1f}% of detected regions are water")
        
        return water_mask, ndwi, mndwi
    
    def mask_glaciers(self, bands, water_mask):
        """
        Step 3: Remove glacier/snow regions using NDSI
        """
        print("\n[STEP 3] Glacier Masking (NDSI)...")
        
        green = bands['green']
        swir = bands['swir']
        nir = bands['nir']
        
        eps = 1e-10
        
        # Calculate NDSI
        ndsi = (green - swir) / (green + swir + eps)
        
        # Brightness
        brightness = (bands['blue'] + green + bands['red']) / 3.0
        
        # Normalize brightness
        valid = brightness[brightness > 0]
        if len(valid) > 0:
            p2, p98 = np.percentile(valid, [2, 98])
            brightness_norm = np.clip((brightness - p2) / (p98 - p2 + 1e-10), 0, 1)
        else:
            brightness_norm = brightness
        
        # Glacier mask: High NDSI + High brightness + High NIR
        glacier_mask = (
            (ndsi > 0.25) &
            (brightness_norm > 0.6) &
            (nir > np.percentile(nir[nir > 0], 70))
        )
        
        # Remove glaciers from water mask
        lake_mask = water_mask & (~glacier_mask)
        
        removed_pct = (np.sum(water_mask) - np.sum(lake_mask)) / np.sum(water_mask) * 100 if np.sum(water_mask) > 0 else 0
        print(f"  Removed {removed_pct:.1f}% as glacier/snow")
        
        return lake_mask, glacier_mask, ndsi
    
    def classify_lakes(self, bands, lake_mask):
        """
        Step 4: Classify remaining lakes as Glacial vs Non-glacial
        Based on proximity to glaciers and elevation
        """
        print("\n[STEP 4] Lake Classification...")
        
        # For now, simple classification based on elevation/context
        # In full version, this would use glacier proximity from RGI/GLIMS
        
        # Label connected components
        labeled_lakes, num_lakes = label(lake_mask)
        
        glacial_lakes = np.zeros_like(lake_mask, dtype=bool)
        non_glacial_lakes = np.zeros_like(lake_mask, dtype=bool)
        
        # Simple heuristic: lakes in upper part of image are more likely glacial
        h = lake_mask.shape[0]
        upper_threshold = h * 0.6  # Upper 60% of image
        
        for i in range(1, num_lakes + 1):
            lake_pixels = (labeled_lakes == i)
            
            # Get centroid
            y_coords, x_coords = np.where(lake_pixels)
            centroid_y = np.mean(y_coords)
            
            # Classify based on position (placeholder logic)
            if centroid_y < upper_threshold:
                glacial_lakes |= lake_pixels
            else:
                non_glacial_lakes |= lake_pixels
        
        print(f"  Classified {np.sum(glacial_lakes > 0)} glacial lakes")
        print(f"  Classified {np.sum(non_glacial_lakes > 0)} non-glacial lakes")
        
        return glacial_lakes, non_glacial_lakes
    
    def create_comprehensive_visualization(self, rgb, bands, detection_mask, water_mask, 
                                          glacier_mask, glacial_lakes, non_glacial_lakes, 
                                          detections, date_str, output_path):
        """Create comprehensive multi-panel visualization"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Original RGB
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(rgb)
        ax1.set_title('1. Original RGB (Enhanced)', fontweight='bold', fontsize=11)
        ax1.axis('off')
        
        # 2. YOLO Detections
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(rgb)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                           edgecolor='yellow', facecolor='none')
            ax2.add_patch(rect)
            ax2.text(x1, y1-5, f"{det['confidence']:.2f}", 
                    color='yellow', fontsize=8, fontweight='bold')
        ax2.set_title(f'2. YOLO Detections ({len(detections)} regions)', fontweight='bold', fontsize=11)
        ax2.axis('off')
        
        # 3. Water Verification
        ax3 = fig.add_subplot(gs[0, 2])
        overlay = rgb.copy()
        overlay[water_mask] = [0, 255, 255]  # Cyan for water
        ax3.imshow(cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0))
        ax3.set_title('3. Water Verification (NDWI/MNDWI)', fontweight='bold', fontsize=11)
        ax3.axis('off')
        
        # 4. Glacier Mask
        ax4 = fig.add_subplot(gs[1, 0])
        glacier_viz = np.zeros_like(rgb)
        glacier_viz[glacier_mask] = [200, 200, 200]  # Grey for glaciers
        ax4.imshow(cv2.addWeighted(rgb, 0.5, glacier_viz, 0.5, 0))
        ax4.set_title('4. Glacier Mask (NDSI)', fontweight='bold', fontsize=11)
        ax4.axis('off')
        
        # 5. Final Classification
        ax5 = fig.add_subplot(gs[1, 1])
        final_viz = rgb.copy()
        final_viz[glacial_lakes] = [0, 191, 255]  # Deep sky blue for glacial lakes
        final_viz[non_glacial_lakes] = [255, 140, 0]  # Dark orange for non-glacial
        ax5.imshow(final_viz)
        ax5.set_title('5. Final Classification', fontweight='bold', fontsize=11)
        ax5.axis('off')
        
        # 6. Classification Map
        ax6 = fig.add_subplot(gs[1, 2])
        class_map = np.zeros(rgb.shape[:2], dtype=np.uint8)
        class_map[glacier_mask] = 1  # Glacier
        class_map[glacial_lakes] = 2  # Glacial Lake
        class_map[non_glacial_lakes] = 3  # Non-glacial Lake
        
        colors = ['#8B4513', '#D3D3D3', '#00BFFF', '#FF8C00']
        cmap = ListedColormap(colors)
        ax6.imshow(class_map, cmap=cmap, vmin=0, vmax=3)
        ax6.set_title('6. Classification Map', fontweight='bold', fontsize=11)
        ax6.axis('off')
        
        # Legend
        legend_elements = [
            Patch(facecolor='#8B4513', label='Non-Glacier'),
            Patch(facecolor='#D3D3D3', label='Glacier'),
            Patch(facecolor='#00BFFF', label='Glacial Lake'),
            Patch(facecolor='#FF8C00', label='Non-Glacial Lake')
        ]
        ax6.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # 7-9. Spectral Indices
        indices = [
            (bands['green'] - bands['nir']) / (bands['green'] + bands['nir'] + 1e-10),  # NDWI
            (bands['green'] - bands['swir']) / (bands['green'] + bands['swir'] + 1e-10),  # MNDWI
            (bands['green'] - bands['swir']) / (bands['green'] + bands['swir'] + 1e-10)  # NDSI
        ]
        titles = ['7. NDWI', '8. MNDWI', '9. NDSI']
        
        for idx, (ax_idx, index, title) in enumerate(zip([gs[2, 0], gs[2, 1], gs[2, 2]], indices, titles)):
            ax = fig.add_subplot(ax_idx)
            im = ax.imshow(index, cmap='RdYlBu', vmin=-0.5, vmax=0.5)
            ax.set_title(title, fontweight='bold', fontsize=11)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Overall title
        fig.suptitle(f'YOLO-Enhanced Glacial Lake Detection Pipeline - {date_str}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"\n✓ Saved comprehensive visualization: {output_path}")


def test_single_image(tiff_path, output_dir):
    """Test the pipeline on a single image"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    date_str = os.path.basename(tiff_path)[10:20]
    
    print(f"\n{'='*70}")
    print(f"YOLO-ENHANCED GLACIAL LAKE DETECTION - SINGLE IMAGE TEST")
    print(f"{'='*70}")
    print(f"Image: {os.path.basename(tiff_path)}")
    print(f"Date: {date_str}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")
    
    # Initialize detector
    detector = YOLOGlacialLakeDetector(model_size='n')  # Use nano for speed
    
    # Load bands
    print("\nLoading and enhancing bands...")
    bands = detector.load_and_enhance_bands(tiff_path)
    
    # Create RGB for YOLO
    print("Creating RGB composite for YOLO...")
    rgb = detector.create_rgb_for_yolo(bands)
    
    # Run 4-step pipeline
    detection_mask, detections = detector.detect_with_yolo(rgb)
    water_mask, ndwi, mndwi = detector.verify_water_with_indices(bands, detection_mask)
    lake_mask, glacier_mask, ndsi = detector.mask_glaciers(bands, water_mask)
    glacial_lakes, non_glacial_lakes = detector.classify_lakes(bands, lake_mask)
    
    # Create visualization
    output_path = os.path.join(output_dir, f"YOLO_Detection_{date_str}.png")
    detector.create_comprehensive_visualization(
        rgb, bands, detection_mask, water_mask, glacier_mask,
        glacial_lakes, non_glacial_lakes, detections, date_str, output_path
    )
    
    print(f"\n{'='*70}")
    print("TEST COMPLETE!")
    print(f"{'='*70}\n")
    
    return output_path


if __name__ == "__main__":
    # Test on first available image
    TIFF_DIR = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\yolo_test"
    
    import glob
    tiff_files = sorted(glob.glob(os.path.join(TIFF_DIR, "*.tif")))
    
    if tiff_files:
        # Test on first image
        test_image = tiff_files[0]
        result = test_single_image(test_image, OUTPUT_DIR)
        print(f"Check the result: {result}")
    else:
        print("No TIFF files found!")
