"""
Sentinel-1 Lake Detection using Ollama Vision (LLaVA)
======================================================
Uses LLaVA vision model through Ollama for intelligent lake detection.

The model analyzes SAR imagery and identifies:
- Glacial lakes (dark smooth areas)
- Shadows vs actual water bodies
- Glacier boundaries

Author: Antigravity
Date: 2026-01-30
"""

import os
import base64
import json
import requests
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import uniform_filter, median_filter
import warnings

warnings.filterwarnings('ignore')

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"


def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def query_ollama_vision(image_path, prompt, model="llava"):
    """
    Send image to Ollama vision model for analysis.
    
    Parameters:
        image_path: Path to PNG/JPG image
        prompt: Question/instruction for the model
        model: Ollama model name (llava, llava:13b, etc.)
    
    Returns:
        Model's response text
    """
    image_b64 = image_to_base64(image_path)
    
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama: {e}")
        return None


def load_and_prepare_sar_image(tiff_path, output_png):
    """Load SAR data and create enhanced PNG for vision model"""
    with rasterio.open(tiff_path) as src:
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)
        profile = src.profile.copy()
    
    # Denoise
    vv = median_filter(vv, size=3)
    
    # Enhanced Lee filter
    vv_mean = uniform_filter(vv.astype(np.float64), 5)
    vv_sqr_mean = uniform_filter(vv.astype(np.float64)**2, 5)
    vv_var = np.maximum(vv_sqr_mean - vv_mean**2, 0)
    overall_var = np.var(vv)
    w = vv_var / (vv_var + overall_var + 1e-10)
    vv_denoised = vv_mean + w * (vv - vv_mean)
    
    # Stretch to 0-255
    p1 = np.percentile(vv_denoised, 1)
    p99 = np.percentile(vv_denoised, 99)
    stretched = np.clip((vv_denoised - p1) / (p99 - p1 + 1e-10), 0, 1)
    img_8bit = (stretched * 255).astype(np.uint8)
    
    # Save PNG
    img = Image.fromarray(img_8bit, mode='L')
    img.save(output_png)
    
    return vv, vh, profile, output_png


def detect_lakes_with_vision(image_path, model="llava"):
    """
    Use Ollama Vision to analyze SAR image and describe lake locations.
    """
    prompts = [
        # Initial analysis
        """Analyze this Synthetic Aperture Radar (SAR) satellite image. 
        This is a grayscale SAR image where:
        - Dark areas (black/very dark gray) = smooth surfaces (water bodies, glacial lakes)
        - Bright areas (white/light gray) = rough surfaces (glaciers, rocky terrain)
        - Medium gray = various terrain types
        
        Please identify and describe:
        1. Where are the water bodies/lakes in this image? (describe their location: top-left, center, bottom-right, etc.)
        2. How many distinct lake/water bodies can you see?
        3. Are the dark areas actual lakes or shadows? (lakes are usually very uniform/smooth, shadows may have some texture)
        
        Be specific about locations using image quadrants.""",
        
        # Detailed lake analysis
        """Looking at this SAR image, focus on the DARK areas.
        For each dark region you can identify:
        1. Is it likely a glacial lake (very smooth, uniform dark) or a shadow (some texture visible)?
        2. What is its approximate shape (elongated, round, irregular)?
        3. Where is it located in the image?
        
        List each potential lake with its characteristics.""",
        
        # Classification guidance
        """Based on your analysis of this SAR image, provide classification suggestions:
        1. Which specific regions should be classified as WATER/LAKE?
        2. Which dark regions are likely SHADOWS (not water)?
        3. Which bright regions are GLACIERS?
        4. What percentage of the image appears to be glacial lakes?
        
        Give specific location descriptions for each class."""
    ]
    
    responses = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n  Analyzing with prompt {i+1}/3...")
        response = query_ollama_vision(image_path, prompt, model)
        if response:
            responses.append(response)
            print(f"  Response received ({len(response)} chars)")
        else:
            print(f"  No response received")
    
    return responses


def create_analysis_report(responses, output_path):
    """Save vision model responses as a report"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("OLLAMA VISION LAKE DETECTION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        for i, response in enumerate(responses):
            f.write(f"\n--- Analysis {i+1} ---\n\n")
            f.write(response)
            f.write("\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"  [OK] Saved report: {output_path}")


def run_vision_lake_detection(tiff_path, output_dir, model="llava"):
    """
    Complete pipeline: Load SAR -> Prepare PNG -> Analyze with Vision -> Report
    """
    filename = os.path.basename(tiff_path)
    date_str = filename.replace('.tif', '').replace('S1_', '')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"OLLAMA VISION LAKE DETECTION: {date_str}")
    print(f"Model: {model}")
    print("=" * 70)
    
    # 1. Prepare image
    print("\n[1] Preparing SAR image for vision model...")
    png_path = os.path.join(output_dir, f"S1_ForVision_{date_str}.png")
    vv, vh, profile, png_path = load_and_prepare_sar_image(tiff_path, png_path)
    print(f"    Saved: {png_path}")
    
    # 2. Run vision analysis
    print("\n[2] Running Ollama Vision analysis...")
    responses = detect_lakes_with_vision(png_path, model)
    
    if responses:
        # 3. Create report
        print("\n[3] Creating analysis report...")
        report_path = os.path.join(output_dir, f"S1_VisionAnalysis_{date_str}.txt")
        create_analysis_report(responses, report_path)
        
        # Print summary to console
        print("\n" + "=" * 70)
        print("VISION MODEL ANALYSIS SUMMARY:")
        print("=" * 70)
        for i, resp in enumerate(responses):
            print(f"\n--- Analysis {i+1} ---")
            # Print first 500 chars of each response
            print(resp[:800] + "..." if len(resp) > 800 else resp)
    else:
        print("\n  ERROR: No responses from vision model")
        print("  Make sure Ollama is running and llava model is installed:")
        print("    ollama pull llava")
        print("    ollama serve")
    
    print("\n" + "=" * 70)
    print("VISION ANALYSIS COMPLETE!")
    print(f"Output folder: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    TIFF_PATH = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1\S1_2023-07-18.tif"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_vision_detection"
    
    run_vision_lake_detection(TIFF_PATH, OUTPUT_DIR, model="llava")
