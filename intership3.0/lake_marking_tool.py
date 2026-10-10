"""
Simple Lake Marking Tool
=========================
Draw directly on the image to mark classes:
- Press 'L' then draw = Mark LAKE (blue)
- Press 'G' then draw = Mark GLACIER (white)  
- Press 'T' then draw = Mark TERRAIN (brown)
- Press 'E' = Erase mode
- Press 'S' = SAVE and classify
- Press 'Q'
 = Quit without saving

Controls:
- Click and drag to paint
- Scroll mouse wheel to change brush size
- Press keys to change mode

Author: Antigravity
Date: 2026-01-23
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage import exposure
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


class LakeMarkingTool:
    def __init__(self, tiff_path):
        self.tiff_path = tiff_path
        self.mask = None
        self.rgb_display = None
        self.brush_size = 10
        self.current_color = 'lake'  # 'lake', 'glacier', 'terrain', 'erase'
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        
        # Color mapping
        self.colors = {
            'lake': (0, 100, 255),      # Blue
            'glacier': (255, 255, 255),  # White
            'terrain': (139, 69, 19),    # Brown
            'erase': (128, 128, 128)     # Grey (will be transparent)
        }
        
        print(f"\n{'='*70}")
        print("LAKE MARKING TOOL")
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
        
        # Scale to 0-1
        scale = 10000.0 if np.max(blue) > 2.0 else 1.0
        b = np.clip(blue / scale, 0, 1)
        g = np.clip(green / scale, 0, 1)
        r = np.clip(red / scale, 0, 1)
        
        # Create RGB
        rgb = np.dstack([r, g, b])
        
        # Enhance for display
        self.rgb_display = exposure.equalize_adapthist(rgb, clip_limit=0.03)
        
        # Convert to 0-255 for drawing
        self.rgb_display = (self.rgb_display * 255).astype(np.uint8)
        
        # Initialize transparent mask (RGBA)
        h, w = self.rgb_display.shape[:2]
        self.mask = np.zeros((h, w, 4), dtype=np.uint8)
        self.mask[:, :, 3] = 0  # Fully transparent initially
        
        print(f"  Image loaded: {w} x {h}")
        
    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'l':
            self.current_color = 'lake'
            print("Mode: LAKE (blue) - Click and drag to mark lakes")
        elif event.key == 'g':
            self.current_color = 'glacier'
            print("Mode: GLACIER (white) - Click and drag to mark glaciers")
        elif event.key == 't':
            self.current_color = 'terrain'
            print("Mode: TERRAIN (brown) - Click and drag to mark terrain/shadows")
        elif event.key == 'e':
            self.current_color = 'erase'
            print("Mode: ERASE - Click and drag to erase marks")
        elif event.key == 's':
            print("\nSaving and running classification...")
            self.save_and_classify()
        elif event.key == 'q':
            print("\nQuitting without saving...")
            plt.close()
    
    def on_scroll(self, event):
        """Handle mouse scroll for brush size"""
        if event.button == 'up':
            self.brush_size = min(50, self.brush_size + 2)
        elif event.button == 'down':
            self.brush_size = max(3, self.brush_size - 2)
        print(f"Brush size: {self.brush_size}")
    
    def on_press(self, event):
        """Mouse button pressed"""
        if event.xdata is None or event.ydata is None:
            return
        self.is_drawing = True
        self.draw_at(int(event.xdata), int(event.ydata))
    
    def on_release(self, event):
        """Mouse button released"""
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
    
    def on_motion(self, event):
        """Mouse moved"""
        if not self.is_drawing or event.xdata is None or event.ydata is None:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        # Draw line from last point to current (for smooth drawing)
        if self.last_x is not None:
            self.draw_line(self.last_x, self.last_y, x, y)
        else:
            self.draw_at(x, y)
        
        self.last_x = x
        self.last_y = y
    
    def draw_line(self, x0, y0, x1, y1):
        """Draw line between two points (for smooth continuous drawing)"""
        # Bresenham's line algorithm
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        # Draw at each point
        for x, y in points:
            self.draw_at(x, y)
    
    def draw_at(self, x, y):
        """Draw at specific location"""
        h, w = self.mask.shape[:2]
        
        # Get color
        if self.current_color == 'erase':
            color = (0, 0, 0, 0)  # Transparent
        else:
            r, g, b = self.colors[self.current_color]
            color = (r, g, b, 180)  # Semi-transparent
        
        # Draw circle
        for dy in range(-self.brush_size, self.brush_size + 1):
            for dx in range(-self.brush_size, self.brush_size + 1):
                if dx*dx + dy*dy <= self.brush_size*self.brush_size:
                    px = x + dx
                    py = y + dy
                    if 0 <= px < w and 0 <= py < h:
                        if self.current_color == 'erase':
                            self.mask[py, px] = [0, 0, 0, 0]
                        else:
                            self.mask[py, px] = color
        
        # Update display
        self.update_display()
    
    def update_display(self):
        """Update the displayed image"""
        # Blend mask with original image
        combined = self.rgb_display.copy()
        
        # Alpha blend where mask is not transparent
        alpha = self.mask[:, :, 3:4] / 255.0
        combined = (1 - alpha) * combined + alpha * self.mask[:, :, :3]
        combined = combined.astype(np.uint8)
        
        self.ax.clear()
        self.ax.imshow(combined)
        self.ax.set_title(f'Mode: {self.current_color.upper()} | Brush: {self.brush_size} | ' +
                         'L=Lake G=Glacier T=Terrain E=Erase S=Save Q=Quit',
                         fontsize=11, fontweight='bold')
        self.ax.axis('off')
        self.fig.canvas.draw_idle()
    
    def save_and_classify(self):
        """Save mask and run classification"""
        date_str = os.path.basename(self.tiff_path)[10:20]
        output_dir = r"f:\NIDM-SRM Drone Project Docs\intership3.0\mask_based_classification"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save mask as PNG
        mask_path = os.path.join(output_dir, f"hand_marked_mask_{date_str}.png")
        mask_rgb = self.mask[:, :, :3]  # Remove alpha channel
        Image.fromarray(mask_rgb).save(mask_path)
        print(f"\n✓ Saved mask: {mask_path}")
        
        plt.close()
        
        # Run classification
        print("\nRunning classification from your markings...")
        from mask_based_classifier import run_mask_based_classification
        run_mask_based_classification(self.tiff_path, mask_path, output_dir)
    
    def run(self):
        """Start the interactive tool"""
        print(f"\n{'='*70}")
        print("CONTROLS:")
        print("  L = Lake mode (blue)")
        print("  G = Glacier mode (white)")
        print("  T = Terrain mode (brown)")
        print("  E = Erase mode")
        print("  Mouse wheel = Change brush size")
        print("  S = SAVE and run classification")
        print("  Q = Quit")
        print(f"{'='*70}\n")
        print("Starting in LAKE mode (blue)...")
        print("Click and drag to mark the major lakes!\n")
        
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.update_display()
        
        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    TIFF_PATH = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\Sentinel2_2021-11-08.tif"
    
    if not os.path.exists(TIFF_PATH):
        print(f"ERROR: File not found: {TIFF_PATH}")
        print("Please update TIFF_PATH in the script.")
    else:
        tool = LakeMarkingTool(TIFF_PATH)
        tool.run()
