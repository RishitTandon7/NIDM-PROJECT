"""
Web-Based Lake Marking Tool
============================
Opens in your browser for smooth, fast drawing!

Author: Antigravity
Date: 2026-01-23
"""

import os
import numpy as np
import rasterio
from skimage import exposure
from PIL import Image
import base64
from io import BytesIO
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import warnings

warnings.filterwarnings('ignore')


class LakeMarkingServer:
    def __init__(self, tiff_path):
        self.tiff_path = tiff_path
        self.output_dir = r"f:\NIDM-SRM Drone Project Docs\intership3.0\mask_based_classification"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("WEB-BASED LAKE MARKING TOOL")
        print(f"{'='*70}")
        print("Loading image...")
        
        self.load_image()
        self.create_html()
        
    def load_image(self):
        """Load and prepare Sentinel-2 image"""
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
        
        # Enhance
        rgb_enhanced = exposure.equalize_adapthist(rgb, clip_limit=0.03)
        rgb_enhanced = (rgb_enhanced * 255).astype(np.uint8)
        
        # Scale down for web display
        max_size = 1400
        scale_factor = min(max_size / self.width, max_size / self.height, 1.0)
        
        if scale_factor < 1.0:
            from skimage.transform import resize
            new_h = int(self.height * scale_factor)
            new_w = int(self.width * scale_factor)
            rgb_enhanced = (resize(rgb_enhanced, (new_h, new_w), anti_aliasing=True) * 255).astype(np.uint8)
            self.display_scale = scale_factor
            print(f"  Original: {self.width} x {self.height}")
            print(f"  Display: {new_w} x {new_h}")
        else:
            self.display_scale = 1.0
            print(f"  Image size: {self.width} x {self.height}")
        
        # Convert to base64 for web
        img = Image.fromarray(rgb_enhanced)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        self.image_base64 = base64.b64encode(buffer.getvalue()).decode()
        self.display_width, self.display_height = img.size
        
        print("  Image encoded for web display")
        
    def create_html(self):
        """Create HTML interface"""
        self.html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Lake Marking Tool</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: white;
            font-family: Arial, sans-serif;
            overflow-x: hidden;
        }}
        #container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #4CAF50;
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
        .btn:hover {{
            transform: scale(1.05);
        }}
        .btn.lake {{ background: #2196F3; color: white; }}
        .btn.glacier {{ background: white; color: black; }}
        .btn.terrain {{ background: #8B4513; color: white; }}
        .btn.erase {{ background: #666; color: white; }}
        .btn.active {{
            box-shadow: 0 0 15px rgba(255,255,255,0.5);
        }}
        #save {{ background: #4CAF50; color: white; padding: 12px 30px; font-size: 16px; }}
        #clear {{ background: #f44336; color: white; }}
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
    </style>
</head>
<body>
    <div id="container">
        <h1>🏔️ Lake Marking Tool</h1>
        
        <div id="controls">
            <button class="btn lake active" onclick="setMode('lake')">1 - Lake (Blue)</button>
            <button class="btn glacier" onclick="setMode('glacier')">2 - Glacier (White)</button>
            <button class="btn terrain" onclick="setMode('terrain')">3 - Terrain (Brown)</button>
            <button class="btn erase" onclick="setMode('erase')">4 - Erase</button>
            <span style="margin-left: 20px;">Brush Size:</span>
            <input type="range" id="brushSize" min="5" max="50" value="15" oninput="updateBrushSize()">
            <span id="brushValue">15px</span>
            <button class="btn clear" id="clear" onclick="clearMask()">Clear All</button>
            <button class="btn" id="save" onclick="saveAndClassify()">💾 Save & Classify</button>
        </div>
        
        <div id="canvasContainer">
            <canvas id="baseCanvas"></canvas>
            <canvas id="drawCanvas"></canvas>
        </div>
        
        <div id="status">
            Click and drag to mark areas. Press number keys 1-4 to change modes. Press S to save.
        </div>
    </div>
    
    <script>
        // Setup
        const baseCanvas = document.getElementById('baseCanvas');
        const drawCanvas = document.getElementById('drawCanvas');
        const baseCtx = baseCanvas.getContext('2d');
        const drawCtx = drawCanvas.getContext('2d');
        
        const displayWidth = {self.display_width};
        const displayHeight = {self.display_height};
        const displayScale = {self.display_scale};
        
        baseCanvas.width = drawCanvas.width = displayWidth;
        baseCanvas.height = drawCanvas.height = displayHeight;
        
        // Position draw canvas on top
        drawCanvas.style.position = 'absolute';
        drawCanvas.style.left = '0';
        drawCanvas.style.top = '0';
        
        // Load base image
        const img = new Image();
        img.onload = function() {{
            baseCtx.drawImage(img, 0, 0);
        }};
        img.src = 'data:image/png;base64,{self.image_base64}';
        
        // State
        let currentMode = 'lake';
        let brushSize = 15;
        let isDrawing = false;
        let lastX = 0, lastY = 0;
        
        const colors = {{
            lake: '#0064FF',
            glacier: '#FFFFFF',
            terrain: '#8B4513',
            erase: null
        }};
        
        // Functions
        function setMode(mode) {{
            currentMode = mode;
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            document.querySelector('.btn.' + mode).classList.add('active');
            updateStatus();
        }}
        
        function updateBrushSize() {{
            brushSize = parseInt(document.getElementById('brushSize').value);
            document.getElementById('brushValue').textContent = brushSize + 'px';
        }}
        
        function updateStatus() {{
            document.getElementById('status').innerHTML = 
                '<strong>Mode: ' + currentMode.toUpperCase() + '</strong> | Brush: ' + brushSize + 'px | ' +
                'Click and drag to draw. Use keyboard: 1-4 for modes, S to save';
        }}
        
        function draw(x, y) {{
            if (currentMode === 'erase') {{
                drawCtx.globalCompositeOperation = 'destination-out';
                drawCtx.fillStyle = 'rgba(0,0,0,1)';
            }} else {{
                drawCtx.globalCompositeOperation = 'source-over';
                drawCtx.fillStyle = colors[currentMode] + 'BB';  // Semi-transparent
            }}
            
            drawCtx.beginPath();
            drawCtx.arc(x, y, brushSize, 0, Math.PI * 2);
            drawCtx.fill();
        }}
        
        function clearMask() {{
            if (confirm('Clear all markings?')) {{
                drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            }}
        }}
        
        function saveAndClassify() {{
            // Get mask data
            const maskData = drawCanvas.toDataURL('image/png');
            
            // Send to server
            fetch('/save', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{
                    mask: maskData,
                    scale: displayScale
                }})
            }})
            .then(response => response.json())
            .then(data => {{
                alert('✓ Saved! Now running classification...\\n\\nCheck terminal for progress.');
            }})
            .catch(err => {{
                alert('Error: ' + err);
            }});
        }}
        
        // Mouse events
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
            
            // Draw line from last point (smooth)
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
        
        drawCanvas.addEventListener('mouseup', () => {{
            isDrawing = false;
        }});
        
        drawCanvas.addEventListener('mouseleave', () => {{
            isDrawing = false;
        }});
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.key === '1') setMode('lake');
            else if (e.key === '2') setMode('glacier');
            else if (e.key === '3') setMode('terrain');
            else if (e.key === '4') setMode('erase');
            else if (e.key === 's' || e.key === 'S') saveAndClassify();
        }});
        
        updateStatus();
    </script>
</body>
</html>
"""
    
    def save_mask(self, mask_data_url, scale):
        """Save the drawn mask"""
        # Extract base64 data
        mask_data = mask_data_url.split(',')[1]
        mask_bytes = base64.b64decode(mask_data)
        
        # Load as image
        mask_img = Image.open(BytesIO(mask_bytes))
        
        # Scale back to original size if needed
        if scale < 1.0:
            original_size = (self.width, self.height)
            mask_img = mask_img.resize(original_size, Image.Resampling.NEAREST)
        
        # Save
        date_str = os.path.basename(self.tiff_path)[10:20]
        mask_path = os.path.join(self.output_dir, f"hand_marked_mask_{date_str}.png")
        mask_img.save(mask_path)
        print(f"\n✓ Saved mask: {mask_path}")
        
        return mask_path


class RequestHandler(BaseHTTPRequestHandler):
    server_instance = None
    
    def log_message(self, format, *args):
        pass  # Suppress logs
    
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(self.server_instance.html.encode())
    
    def do_POST(self):
        if self.path == '/save':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            # Save mask
            mask_path = self.server_instance.save_mask(data['mask'], data['scale'])
            
            # Run classification
            print("\n Running classification...")
            from mask_based_classifier import run_mask_based_classification
            run_mask_based_classification(
                self.server_instance.tiff_path,
                mask_path,
                self.server_instance.output_dir
            )
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'success'}).encode())


def run_server(tiff_path, port=8765):
    """Run the web server"""
    server_instance = LakeMarkingServer(tiff_path)
    RequestHandler.server_instance = server_instance
    
    server = HTTPServer(('localhost', port), RequestHandler)
    server.server_instance = server_instance
    
    url = f'http://localhost:{port}'
    print(f"\n{'='*70}")
    print(f"Server started at: {url}")
    print(f"{'='*70}")
    print("\nOpening in browser...")
    
    # Open browser
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    
    print("\nServer running. Press Ctrl+C to stop.\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    TIFF_PATH = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\Sentinel2_2021-11-08.tif"
    
    if not os.path.exists(TIFF_PATH):
        print(f"ERROR: File not found: {TIFF_PATH}")
    else:
        run_server(TIFF_PATH)
