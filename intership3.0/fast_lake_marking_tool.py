"""
Fast Lake Marking Tool (OpenCV Version)
=========================================
Much faster drawing performance using OpenCV!

Draw directly on the image to mark classes:
- Press '1' = Mark LAKE (blue)
- Press '2' = Mark GLACIER (white)
- Press '3' = Mark TERRAIN (brown)
- Press '4' = Erase mode
- Press '+'/'-' = Increase/Decrease brush size
- Press 'S' = SAVE and classify
- Press 'Q' = Quit without saving

Controls:
- Click and drag to paint (smooth and fast!)
- Much smoother than matplotlib version

Author: Antigravity
Date: 2026-01-23
"""

import os
import numpy as np
import rasterio
import cv2
from skimage import exposure
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


class FastLakeMarkingTool:
    def __init__(self, tiff_path, max_display_size=1200):
        self.tiff_path = tiff_path
        self.max_display_size = max_display_size
        self.brush_size = 15
        self.current_color = 'lake'
        self.is_drawing = False
        
        # Color mapping (BGR for OpenCV)
        self.colors = {
            'lake': (255, 100, 0),      # Blue (BGR)
            'glacier': (255, 255, 255),  # White
            'terrain': (19, 69, 139),    # Brown
            'erase': None                # Special case
        }
        
        print(f"\n{'='*70}")
        print("FAST LAKE MARKING TOOL (OpenCV)")
        print(f"{'='*70}")
        print("Loading image...")
        self.load_image()
        
    def load_image(self):
        """Load and prepare Sentinel-2 image for display"""
        with rasterio.open(self.tiff_path) as src:
            self.profile = src.profile
            blue = src.read(1).astype(np.float32)
            green = src.read(2).astype(np.float32)
            red = src.read(3).astype(np.float32)
            self.height, self.width = blue.shape
        
        # Scale to 0-1
        scale = 10000.0 if np.max(blue) > 2.0 else 1.0
        b = np.clip(blue / scale, 0, 1)
        g = np.clip(green / scale, 0, 1)
        r = np.clip(red / scale, 0, 1)
        
        # Create RGB
        rgb = np.dstack([r, g, b])
        
        # Enhance for display
        rgb_enhanced = exposure.equalize_adapthist(rgb, clip_limit=0.03)
        rgb_enhanced = (rgb_enhanced * 255).astype(np.uint8)
        
        # Convert to BGR for OpenCV
        self.base_image = cv2.cvtColor(rgb_enhanced, cv2.COLOR_RGB2BGR)
        
        # Calculate display scale
        scale_factor = min(self.max_display_size / self.width, 
                          self.max_display_size / self.height, 1.0)
        
        if scale_factor < 1.0:
            new_w = int(self.width * scale_factor)
            new_h = int(self.height * scale_factor)
            self.display_image = cv2.resize(self.base_image, (new_w, new_h))
            self.scale_factor = scale_factor
            print(f"  Original: {self.width} x {self.height}")
            print(f"  Display: {new_w} x {new_h} (scaled for speed)")
        else:
            self.display_image = self.base_image.copy()
            self.scale_factor = 1.0
            print(f"  Image size: {self.width} x {self.height}")
        
        # Create mask (same size as display image)
        self.mask = np.zeros(self.display_image.shape, dtype=np.uint8)
        
        # For saving (full resolution mask)
        self.full_mask = np.zeros(self.base_image.shape, dtype=np.uint8)
        
        # Create display
        self.update_display()
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.draw_at(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            self.draw_at(x, y)
    
    def draw_at(self, x, y):
        """Draw at specific location (much faster than matplotlib!)"""
        if self.current_color == 'erase':
            # Erase by restoring base image
            cv2.circle(self.mask, (x, y), self.brush_size, (0, 0, 0), -1)
            # Also erase from full resolution mask
            full_x = int(x / self.scale_factor)
            full_y = int(y / self.scale_factor)
            full_brush = int(self.brush_size / self.scale_factor)
            cv2.circle(self.full_mask, (full_x, full_y), full_brush, (0, 0, 0), -1)
        else:
            color = self.colors[self.current_color]
            # Draw on display mask
            cv2.circle(self.mask, (x, y), self.brush_size, color, -1)
            # Draw on full resolution mask
            full_x = int(x / self.scale_factor)
            full_y = int(y / self.scale_factor)
            full_brush = int(self.brush_size / self.scale_factor)
            cv2.circle(self.full_mask, (full_x, full_y), full_brush, color, -1)
        
        self.update_display()
    
    def update_display(self):
        """Update the displayed image (FAST!)"""
        # Blend mask with base image
        mask_alpha = (self.mask != 0).any(axis=2).astype(np.uint8) * 180
        
        # Create blended image
        output = self.display_image.copy()
        for c in range(3):
            output[:, :, c] = np.where(mask_alpha > 0,
                                       (output[:, :, c] * (255 - mask_alpha) + 
                                        self.mask[:, :, c] * mask_alpha) // 255,
                                       output[:, :, c])
        
        # Add status bar
        status_h = 40
        status_bar = np.zeros((status_h, output.shape[1], 3), dtype=np.uint8)
        status_text = (f"Mode: {self.current_color.upper()} | "
                      f"Brush: {self.brush_size} | "
                      f"1=Lake 2=Glacier 3=Terrain 4=Erase +/- S=Save Q=Quit")
        cv2.putText(status_bar, status_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine
        self.current_display = np.vstack([status_bar, output])
        
        cv2.imshow('Lake Marking Tool', self.current_display)
    
    def save_and_classify(self):
        """Save mask and run classification"""
        date_str = os.path.basename(self.tiff_path)[10:20]
        output_dir = r"f:\NIDM-SRM Drone Project Docs\intership3.0\mask_based_classification"
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert full resolution mask from BGR to RGB
        mask_rgb = cv2.cvtColor(self.full_mask, cv2.COLOR_BGR2RGB)
        
        # Save mask as PNG
        mask_path = os.path.join(output_dir, f"hand_marked_mask_{date_str}.png")
        Image.fromarray(mask_rgb).save(mask_path)
        print(f"\n✓ Saved mask: {mask_path}")
        
        cv2.destroyAllWindows()
        
        # Run classification
        print("\nRunning classification from your markings...")
        from mask_based_classifier import run_mask_based_classification
        run_mask_based_classification(self.tiff_path, mask_path, output_dir)
    
    def run(self):
        """Start the interactive tool"""
        print(f"\n{'='*70}")
        print("CONTROLS:")
        print("  1 = Lake mode (blue)")
        print("  2 = Glacier mode (white)")
        print("  3 = Terrain mode (brown)")
        print("  4 = Erase mode")
        print("  +/- = Increase/Decrease brush size")
        print("  S = SAVE and run classification")
        print("  Q = Quit")
        print(f"{'='*70}\n")
        print("Starting in LAKE mode (blue)...")
        print("Click and drag to mark the major lakes!")
        print("This version is MUCH faster than matplotlib!\n")
        
        cv2.namedWindow('Lake Marking Tool')
        cv2.setMouseCallback('Lake Marking Tool', self.mouse_callback)
        
        while True:
            cv2.imshow('Lake Marking Tool', self.current_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('1'):
                self.current_color = 'lake'
                print("Mode: LAKE (blue)")
                self.update_display()
            elif key == ord('2'):
                self.current_color = 'glacier'
                print("Mode: GLACIER (white)")
                self.update_display()
            elif key == ord('3'):
                self.current_color = 'terrain'
                print("Mode: TERRAIN (brown)")
                self.update_display()
            elif key == ord('4'):
                self.current_color = 'erase'
                print("Mode: ERASE")
                self.update_display()
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(50, self.brush_size + 3)
                print(f"Brush size: {self.brush_size}")
                self.update_display()
            elif key == ord('-') or key == ord('_'):
                self.brush_size = max(3, self.brush_size - 3)
                print(f"Brush size: {self.brush_size}")
                self.update_display()
            elif key == ord('s') or key == ord('S'):
                print("\nSaving and running classification...")
                self.save_and_classify()
                break
            elif key == ord('q') or key == ord('Q'):
                print("\nQuitting without saving...")
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    TIFF_PATH = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\Sentinel2_2021-11-08.tif"
    
    if not os.path.exists(TIFF_PATH):
        print(f"ERROR: File not found: {TIFF_PATH}")
        print("Please update TIFF_PATH in the script.")
    else:
        tool = FastLakeMarkingTool(TIFF_PATH)
        tool.run()
