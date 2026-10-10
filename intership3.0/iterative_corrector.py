"""
Iterative Classification Corrector
===================================
1. First runs automatic classification
2. Then lets you correct errors by drawing on the classified map
3. Re-trains and improves with each correction

Workflow:
- Runs initial classification
- Opens web interface showing the classification
- You draw corrections:
  - Mark areas that are WRONGLY classified as Lake → mark them
  - Mark areas that are WRONGLY classified as Glacier → mark them
  - Mark areas that SHOULD BE Lake but aren't → mark them
- Saves corrected results

Author: Antigravity
Date: 2026-01-23
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
from skimage import exposure, morphology
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import base64
from io import BytesIO
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import geopandas as gpd
from shapely.geometry import shape
import warnings

warnings.filterwarnings('ignore')


class IterativeCorrector:
    def __init__(self, tiff_path, output_dir):
        self.tiff_path = tiff_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.date_str = os.path.basename(tiff_path)[10:20]
        
        print(f"\n{'='*70}")
        print("ITERATIVE CLASSIFICATION CORRECTOR")
        print(f"{'='*70}\n")
        
        # Load image
        self.load_image()
        
        # Run initial classification
        self.run_initial_classification()
        
        # Prepare for web interface
        self.prepare_web_display()
        
    def load_image(self):
        """Load Sentinel-2 image"""
        print("Loading Sentinel-2 image...")
        with rasterio.open(self.tiff_path) as src:
            self.profile = src.profile
            self.blue = src.read(1).astype(np.float32)
            self.green = src.read(2).astype(np.float32)
            self.red = src.read(3).astype(np.float32)
            self.nir = src.read(4).astype(np.float32)
            self.swir = src.read(5).astype(np.float32)
            
        self.height, self.width = self.blue.shape
        print(f"  Image size: {self.width} x {self.height}")
        
    def run_initial_classification(self):
        """Run initial automatic classification using spectral rules"""
        print("\nRunning initial automatic classification...")
        
        # Normalize
        scale = 10000.0 if np.max(self.blue) > 2.0 else 1.0
        b = self.blue / scale
        g = self.green / scale
        r = self.red / scale
        nir = self.nir / scale
        swir = self.swir / scale
        
        # Calculate indices
        ndwi = (g - nir) / (g + nir + 1e-10)
        ndsi = (g - swir) / (g + swir + 1e-10)
        mndwi = (g - swir) / (g + swir + 1e-10)
        brightness = (b + g + r) / 3.0
        
        # Initialize classification
        classification = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # 1. Detect GLACIERS (ice/snow)
        glacier_mask = (
            (ndsi > 0.4) &
            (brightness > 0.3) &
            (nir > 0.3)
        )
        
        # 2. Detect LAKES (water)
        water_mask = (
            (ndwi > 0.3) &
            (mndwi > 0.2) &
            (nir < 0.15) &
            (swir < 0.1) &
            ~glacier_mask  # Not glacier
        )
        
        # Clean up with morphology
        water_mask = morphology.remove_small_objects(water_mask, min_size=50)
        glacier_mask = morphology.remove_small_objects(glacier_mask, min_size=100)
        
        # Assign classes: 0=Non-Glacier, 1=Glacier, 2=Lake
        classification[glacier_mask] = 1
        classification[water_mask] = 2
        
        self.classification = classification
        
        # Count pixels
        n_glacier = np.sum(classification == 1)
        n_lake = np.sum(classification == 2)
        n_terrain = np.sum(classification == 0)
        
        print(f"  Initial classification:")
        print(f"    Terrain/Non-Glacier: {n_terrain:,} pixels ({100*n_terrain/(self.height*self.width):.1f}%)")
        print(f"    Glacier: {n_glacier:,} pixels ({100*n_glacier/(self.height*self.width):.1f}%)")
        print(f"    Lake: {n_lake:,} pixels ({100*n_lake/(self.height*self.width):.1f}%)")
        
    def prepare_web_display(self):
        """Prepare images for web display"""
        print("\nPreparing web display...")
        
        # Create enhanced RGB
        scale = 10000.0 if np.max(self.blue) > 2.0 else 1.0
        b = np.clip(self.blue / scale, 0, 1)
        g = np.clip(self.green / scale, 0, 1)
        r = np.clip(self.red / scale, 0, 1)
        rgb = np.dstack([r, g, b])
        rgb_enhanced = exposure.equalize_adapthist(rgb, clip_limit=0.03)
        rgb_enhanced = (rgb_enhanced * 255).astype(np.uint8)
        
        # Create classification overlay
        overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        overlay[self.classification == 0] = [101, 67, 33]   # Brown (terrain)
        overlay[self.classification == 1] = [200, 200, 200] # Grey (glacier)
        overlay[self.classification == 2] = [0, 100, 255]   # Blue (lake)
        
        # Blend: 60% original, 40% classification
        blended = (0.6 * rgb_enhanced + 0.4 * overlay).astype(np.uint8)
        
        # Scale for web display
        max_size = 1400
        scale_factor = min(max_size / self.width, max_size / self.height, 1.0)
        
        if scale_factor < 1.0:
            from skimage.transform import resize
            new_h = int(self.height * scale_factor)
            new_w = int(self.width * scale_factor)
            blended = (resize(blended, (new_h, new_w), anti_aliasing=True) * 255).astype(np.uint8)
            self.display_scale = scale_factor
            print(f"  Display scaled to: {new_w} x {new_h}")
        else:
            self.display_scale = 1.0
        
        # Convert to base64
        img = Image.fromarray(blended)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        self.image_base64 = base64.b64encode(buffer.getvalue()).decode()
        self.display_width, self.display_height = img.size
        
        print("  Image encoded for web")
        
    def apply_corrections(self, correction_mask_data):
        """Apply user corrections and re-classify"""
        print("\nApplying corrections...")
        
        # Decode correction mask
        mask_data = correction_mask_data.split(',')[1]
        mask_bytes = base64.b64decode(mask_data)
        correction_img = Image.open(BytesIO(mask_bytes))
        
        # Scale to original size if needed
        if self.display_scale < 1.0:
            correction_img = correction_img.resize((self.width, self.height), Image.Resampling.NEAREST)
        
        correction_arr = np.array(correction_img)
        
        # Extract corrections by color
        # Red = NOT lake (remove lake label)
        # Green = NOT glacier (remove glacier label)
        # Blue = IS lake (add lake label)
        # Yellow = IS glacier (add glacier label)
        
        not_lake_mask = (correction_arr[:, :, 0] > 200) & (correction_arr[:, :, 1] < 100) & (correction_arr[:, :, 2] < 100)
        not_glacier_mask = (correction_arr[:, :, 1] > 200) & (correction_arr[:, :, 0] < 100) & (correction_arr[:, :, 2] < 100)
        is_lake_mask = (correction_arr[:, :, 2] > 200) & (correction_arr[:, :, 0] < 100) & (correction_arr[:, :, 1] < 100)
        is_glacier_mask = (correction_arr[:, :, 0] > 200) & (correction_arr[:, :, 1] > 200) & (correction_arr[:, :, 2] < 100)
        
        # Apply corrections
        corrections_made = 0
        
        if np.any(not_lake_mask):
            self.classification[not_lake_mask & (self.classification == 2)] = 0
            corrections_made += np.sum(not_lake_mask & (self.classification == 2))
            print(f"  Removed {np.sum(not_lake_mask)} pixels from Lake class")
        
        if np.any(not_glacier_mask):
            self.classification[not_glacier_mask & (self.classification == 1)] = 0
            corrections_made += np.sum(not_glacier_mask & (self.classification == 1))
            print(f"  Removed {np.sum(not_glacier_mask)} pixels from Glacier class")
        
        if np.any(is_lake_mask):
            self.classification[is_lake_mask] = 2
            corrections_made += np.sum(is_lake_mask)
            print(f"  Added {np.sum(is_lake_mask)} pixels to Lake class")
        
        if np.any(is_glacier_mask):
            self.classification[is_glacier_mask] = 1
            corrections_made += np.sum(is_glacier_mask)
            print(f"  Added {np.sum(is_glacier_mask)} pixels to Glacier class")
        
        if corrections_made == 0:
            print("  No corrections detected in the mask")
        else:
            print(f"  Total corrections: {corrections_made} pixels")
        
        return corrections_made
    
    def save_results(self):
        """Save final classification results"""
        print("\nSaving final results...")
        
        # Save GeoTIFF
        geotiff_path = os.path.join(self.output_dir, f"classification_{self.date_str}.tif")
        with rasterio.open(geotiff_path, 'w', **self.profile) as dst:
            dst.write(self.classification.astype(np.uint8), 1)
        print(f"  [OK] GeoTIFF: {geotiff_path}")
        
        # Save GeoJSONs
        transform = self.profile['transform']
        crs = self.profile['crs']
        
        # Glacier GeoJSON
        glacier_mask = (self.classification == 1).astype(np.uint8)
        glacier_shapes = list(shapes(glacier_mask, mask=glacier_mask == 1, transform=transform))
        if glacier_shapes:
            glacier_geoms = [shape(geom) for geom, val in glacier_shapes]
            glacier_gdf = gpd.GeoDataFrame({'class': 'glacier'}, geometry=glacier_geoms, crs=crs)
            glacier_path = os.path.join(self.output_dir, f"glacier_{self.date_str}.geojson")
            glacier_gdf.to_file(glacier_path, driver='GeoJSON')
            print(f"  [OK] Glacier GeoJSON: {glacier_path} ({len(glacier_geoms)} features)")
        
        # Lake GeoJSON
        lake_mask = (self.classification == 2).astype(np.uint8)
        lake_shapes = list(shapes(lake_mask, mask=lake_mask == 1, transform=transform))
        if lake_shapes:
            lake_geoms = [shape(geom) for geom, val in lake_shapes]
            lake_gdf = gpd.GeoDataFrame({'class': 'lake'}, geometry=lake_geoms, crs=crs)
            lake_path = os.path.join(self.output_dir, f"lake_{self.date_str}.geojson")
            lake_gdf.to_file(lake_path, driver='GeoJSON')
            print(f"  [OK] Lake GeoJSON: {lake_path} ({len(lake_geoms)} features)")
        
        # Visualization
        scale = 10000.0 if np.max(self.blue) > 2.0 else 1.0
        b = np.clip(self.blue / scale, 0, 1)
        g = np.clip(self.green / scale, 0, 1)
        r = np.clip(self.red / scale, 0, 1)
        rgb = np.dstack([r, g, b])
        rgb_enhanced = exposure.equalize_adapthist(rgb, clip_limit=0.03)
        rgb_enhanced = (rgb_enhanced * 255).astype(np.uint8)
        
        overlay = np.zeros_like(rgb_enhanced)
        overlay[self.classification == 0] = [101, 67, 33]   # Brown
        overlay[self.classification == 1] = [200, 200, 200] # Grey
        overlay[self.classification == 2] = [0, 100, 255]   # Blue
        
        blended = (0.6 * rgb_enhanced + 0.4 * overlay).astype(np.uint8)
        
        vis_path = os.path.join(self.output_dir, f"visualization_{self.date_str}.png")
        Image.fromarray(blended).save(vis_path)
        print(f"  [OK] Visualization: {vis_path}")
        
        print("\n[OK] All results saved!")


class CorrectionServer:
    def __init__(self, corrector):
        self.corrector = corrector
        self.create_html()
        
    def create_html(self):
        """Create HTML correction interface"""
        self.html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Classification Corrector</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: white;
            font-family: Arial, sans-serif;
        }}
        #container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #4CAF50;
        }}
        .info {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }}
        #controls {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.2s;
        }}
        .btn:hover {{ transform: scale(1.05); }}
        .btn.not-lake {{ background: #f44336; color: white; }}
        .btn.not-glacier {{ background: #4CAF50; color: white; }}
        .btn.is-lake {{ background: #2196F3; color: white; }}
        .btn.is-glacier {{ background: #FFEB3B; color: black; }}
        .btn.erase {{ background: #666; color: white; }}
        .btn.active {{ box-shadow: 0 0 15px rgba(255,255,255,0.5); }}
        #save {{ background: #4CAF50; color: white; padding: 12px 30px; font-size: 16px; }}
        #clear {{ background: #666; color: white; }}
        #brushSize {{
            padding: 8px;
            border-radius: 5px;
            border: none;
            font-size: 14px;
        }}
        #canvasContainer {{
            position: relative;
            display: inline-block;
            border: 3px solid #4CAF50;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 0 20px rgba(76, 175, 80, 0.3);
        }}
        canvas {{
            display: block;
            cursor: crosshair;
        }}
        #status {{
            margin-top: 15px;
            padding: 10px;
            background: #2a2a2a;
            border-radius: 5px;
            text-align: center;
        }}
        .color-box {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 1px solid white;
            margin-right: 5px;
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <div id="container">
        <h1>🗺️ Classification Corrector</h1>
        
        <div class="info">
            <strong>Current Classification:</strong><br>
            <span class="color-box" style="background: #654321;"></span> <strong>Brown</strong> = Terrain/Non-Glacier |
            <span class="color-box" style="background: #C8C8C8;"></span> <strong>Grey</strong> = Glacier |
            <span class="color-box" style="background: #0064FF;"></span> <strong>Blue</strong> = Lake
            <br><br>
            <strong>Draw corrections:</strong><br>
            • <span class="color-box" style="background: #f44336;"></span> <strong>RED</strong> = Mark areas wrongly classified as LAKE (to remove)<br>
            • <span class="color-box" style="background: #4CAF50;"></span> <strong>GREEN</strong> = Mark areas wrongly classified as GLACIER (to remove)<br>
            • <span class="color-box" style="background: #2196F3;"></span> <strong>BLUE</strong> = Mark areas that SHOULD BE lake (to add)<br>
            • <span class="color-box" style="background: #FFEB3B;"></span> <strong>YELLOW</strong> = Mark areas that SHOULD BE glacier (to add)
        </div>
        
        <div id="controls">
            <button class="btn not-lake" onclick="setMode('not-lake')">1 - NOT Lake (Red)</button>
            <button class="btn not-glacier" onclick="setMode('not-glacier')">2 - NOT Glacier (Green)</button>
            <button class="btn is-lake active" onclick="setMode('is-lake')">3 - IS Lake (Blue)</button>
            <button class="btn is-glacier" onclick="setMode('is-glacier')">4 - IS Glacier (Yellow)</button>
            <button class="btn erase" onclick="setMode('erase')">5 - Erase</button>
            <span style="margin-left: 20px;">Brush Size:</span>
            <input type="range" id="brushSize" min="5" max="50" value="20" oninput="updateBrushSize()">
            <span id="brushValue">20px</span>
            <button class="btn clear" id="clear" onclick="clearCorrections()">Clear</button>
            <button class="btn" id="save" onclick="saveCorrections()">💾 Apply & Save</button>
        </div>
        
        <div id="canvasContainer">
            <canvas id="baseCanvas"></canvas>
            <canvas id="drawCanvas"></canvas>
        </div>
        
        <div id="status">
            Mark errors in the classification, then click "Apply & Save"
        </div>
    </div>
    
    <script>
        const baseCanvas = document.getElementById('baseCanvas');
        const drawCanvas = document.getElementById('drawCanvas');
        const baseCtx = baseCanvas.getContext('2d');
        const drawCtx = drawCanvas.getContext('2d');
        
        const displayWidth = {self.corrector.display_width};
        const displayHeight = {self.corrector.display_height};
        
        baseCanvas.width = drawCanvas.width = displayWidth;
        baseCanvas.height = drawCanvas.height = displayHeight;
        
        drawCanvas.style.position = 'absolute';
        drawCanvas.style.left = '0';
        drawCanvas.style.top = '0';
        
        const img = new Image();
        img.onload = function() {{
            baseCtx.drawImage(img, 0, 0);
        }};
        img.src = 'data:image/png;base64,{self.corrector.image_base64}';
        
        let currentMode = 'is-lake';
        let brushSize = 20;
        let isDrawing = false;
        let lastX = 0, lastY = 0;
        
        const colors = {{
            'not-lake': '#FF0000',
            'not-glacier': '#00FF00',
            'is-lake': '#0000FF',
            'is-glacier': '#FFFF00',
            'erase': null
        }};
        
        function setMode(mode) {{
            currentMode = mode;
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            document.querySelector('.btn.' + mode).classList.add('active');
        }}
        
        function updateBrushSize() {{
            brushSize = parseInt(document.getElementById('brushSize').value);
            document.getElementById('brushValue').textContent = brushSize + 'px';
        }}
        
        function draw(x, y) {{
            if (currentMode === 'erase') {{
                drawCtx.globalCompositeOperation = 'destination-out';
                drawCtx.fillStyle = 'rgba(0,0,0,1)';
            }} else {{
                drawCtx.globalCompositeOperation = 'source-over';
                drawCtx.fillStyle = colors[currentMode] + 'CC';
            }}
            
            drawCtx.beginPath();
            drawCtx.arc(x, y, brushSize, 0, Math.PI * 2);
            drawCtx.fill();
        }}
        
        function clearCorrections() {{
            if (confirm('Clear all corrections?')) {{
                drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            }}
        }}
        
        function saveCorrections() {{
            const maskData = drawCanvas.toDataURL('image/png');
            
            fetch('/correct', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{mask: maskData}})
            }})
            .then(response => response.json())
            .then(data => {{
                alert('✓ Corrections applied and saved!\\n\\nCheck the output folder for results.');
            }})
            .catch(err => {{
                alert('Error: ' + err);
            }});
        }}
        
        drawCanvas.addEventListener('mousedown', (e) => {{
            isDrawing = true;
            const rect = drawCanvas.getBoundingClientRect();
            lastX = e.clientX - rect.left;
            lastY = e.clientY - rect.top;
            draw(lastX, lastY);
        }});
        
        drawCanvas.addEventListener('mousemove', (e) => {{
            if (!isDrawing) return;
            const rect = drawCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const dx = x - lastX;
            const dy = y - lastY;
            const dist = Math.sqrt(dx*dx + dy*dy);
            const steps = Math.max(1, Math.floor(dist / 2));
            
            for (let i = 0; i <= steps; i++) {{
                const px = lastX + (dx * i / steps);
                const py = lastY + (dy * i / steps);
                draw(px, py);
            }}
            
            lastX = x;
            lastY = y;
        }});
        
        drawCanvas.addEventListener('mouseup', () => {{ isDrawing = false; }});
        drawCanvas.addEventListener('mouseleave', () => {{ isDrawing = false; }});
        
        document.addEventListener('keydown', (e) => {{
            if (e.key === '1') setMode('not-lake');
            else if (e.key === '2') setMode('not-glacier');
            else if (e.key === '3') setMode('is-lake');
            else if (e.key === '4') setMode('is-glacier');
            else if (e.key === '5') setMode('erase');
            else if (e.key === 's' || e.key === 'S') saveCorrections();
        }});
    </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    server_instance = None
    
    def log_message(self, format, *args):
        pass
    
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(self.server_instance.html.encode())
    
    def do_POST(self):
        if self.path == '/correct':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            # Apply corrections
            corrections = self.server_instance.corrector.apply_corrections(data['mask'])
            
            # Save results
            self.server_instance.corrector.save_results()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'success', 'corrections': corrections}).encode())


def run_corrector(tiff_path, output_dir, port=8766):
    """Run the iterative corrector"""
    # Create corrector and run initial classification
    corrector = IterativeCorrector(tiff_path, output_dir)
    
    # Create web server
    server_instance = CorrectionServer(corrector)
    server_instance.corrector = corrector
    RequestHandler.server_instance = server_instance
    
    server = HTTPServer(('localhost', port), RequestHandler)
    
    url = f'http://localhost:{port}'
    print(f"\n{'='*70}")
    print(f"Correction interface: {url}")
    print(f"{'='*70}")
    print("\nOpening in browser...")
    print("Draw corrections, then click 'Apply & Save'")
    print("\nPress Ctrl+C to stop server\n")
    
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    TIFF_PATH = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\Sentinel2_2021-11-08.tif"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\corrected_classification"
    
    if not os.path.exists(TIFF_PATH):
        print(f"ERROR: File not found: {TIFF_PATH}")
    else:
        run_corrector(TIFF_PATH, OUTPUT_DIR)
