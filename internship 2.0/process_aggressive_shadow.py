"""
Aggressive Shadow Removal using Blue-Red Ratio
Method:
1. Identifies Shadows using "Rayleigh Scattering" signature:
   - Shadows are DARK but relatively BLUE (illuminated by skylight).
   - Rocks are DARK and GRAY/RED.
   - Water is DARK and GREEN/BLUE.
   
2. Logic:
   - IF (Blue > Red * 1.1) AND (Brightness < Low) -> SHADOW (Remove).
   - IF (NDWI > 0.1) AND NOT SHADOW -> LAKE.
   - IF (NDSI > 0) AND NOT SHADOW -> GLACIER.

3. Outputs cleaned visual grid.
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob
from scipy.ndimage import median_filter

def classify_aggressive_shadows(tiff_path):
    with rasterio.open(tiff_path) as src:
        try:
            # 10-band stack: B2, B3, B4, B8... B11(9)
            blue  = src.read(1).astype(float)
            green = src.read(2).astype(float)
            red   = src.read(3).astype(float)
            nir   = src.read(4).astype(float)
            
            # SWIR (Band 9 in 10-band stack, Band 11 normally)
            swir = None
            if src.count >= 9:
                swir = src.read(9).astype(float)
        except:
             return None, None, None

        transform = src.transform
        crs = src.crs

    # --- FEATURE ENGINEERING ---
    np.seterr(divide='ignore', invalid='ignore')
    
    # NDWI (Water)
    ndwi = (green - nir) / (green + nir + 1e-6)
    
    # NDSI (Ice) - Preferred if SWIR exists
    if swir is not None:
        ndsi = (green - swir) / (green + swir + 1e-6)
    else:
        # Fallback NDSI using Red instead of SWIR (modified)
        ndsi = (green - red) / (green + red + 1e-6)
        
    # Brightness (Vis)
    avg_vis = (blue + green + red) / 3.0
    
    # Scale Normalization (Handle 0-1 vs 0-10000)
    max_val = np.nanmax(blue)
    scale_factor = 10000.0 if max_val > 2000 else 1.0
    
    # --- AGGRESSIVE SHADOW DETECTION ---
    
    # Thresholds
    shadow_bright_lim = 0.25 * scale_factor # Pixels brighter than this unlikely to be shadow
    blue_red_ratio_thresh = 1.05 # Shadows are > 5% bluer than red
    
    # 1. SHADOW MASK
    # - Must be Dark
    # - Must have Blue > Red component (Skylight illumination)
    # - Must NOT be Water (Water has very high NDWI)
    
    is_dark = avg_vis < shadow_bright_lim
    is_bluish = blue > (red * blue_red_ratio_thresh)
    
    # Refined Shadow Mask
    # Note: Lakes are also "bluish". We distinguish by NDWI.
    # Lakes have NDWI > 0.2. Shadows often have NDWI ~ 0.0 to 0.1.
    is_shadow = is_dark & is_bluish & (ndwi < 0.2)
    
    # 2. LAKE DETECTOR
    # High NDWI + Dark NIR
    # Must NOT be Shadow
    is_lake = (ndwi > 0.15) & (nir < (0.2 * scale_factor)) & (~is_shadow)
    
    # 3. GLACIER DETECTOR
    # Brightness High OR High NDSI
    # Strict Brightness Check to kill "dark glacier" noise
    glacier_bright_lim = 0.3 * scale_factor
    
    if swir is not None:
        is_glacier = (ndsi > 0.1) & (avg_vis > glacier_bright_lim)
    else:
        is_glacier = (avg_vis > glacier_bright_lim) & ((nir-red)/(nir+red+1e-6) < 0.2)

    # Cleanup
    is_glacier = is_glacier & (~is_lake) & (~is_shadow)
    is_nonglacier = (~is_glacier) & (~is_lake)

    # Filter Speckles (Size < 10 pixels is noise)
    from scipy.ndimage import label, sum as ndi_sum
    def filter_speckles(mask, min_size=10):
        labeled, num = label(mask)
        if num == 0: return mask
        sizes = ndi_sum(mask, labeled, range(num+1))
        mask_size = sizes < min_size
        mask[mask_size[labeled]] = 0
        return mask

    is_glacier = filter_speckles(is_glacier.astype(np.uint8), 15)
    is_lake = filter_speckles(is_lake.astype(np.uint8), 15)
    is_nonglacier = (~is_glacier) & (~is_lake)
    
    return {
        "glacier": is_glacier,
        "glacier_lake": is_lake,
        "nonglacier": is_nonglacier
    }, transform, crs

def process_aggressive(tiff_dir, img_out_dir):
    os.makedirs(img_out_dir, exist_ok=True)
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "Sentinel2_*.tif")))
    
    print(f"Applying Aggressive Shadow Removal to {len(tiff_files)} scenes...")
    
    n_images = len(tiff_files)
    cols = 6
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
    fig.suptitle('Aggressive Blue-Ratio Shadow Removal\nGrey: Glacier | Blue: Lake | Brown: Non-Glacier (Shadows Removed)', fontsize=24, y=0.99)
    axes = axes.flatten()
    cmap = ListedColormap(['#8B4513', '#C0C0C0', '#00BFFF']) # Brown, Silver, Blue

    for i, tiff_path in enumerate(tiff_files):
        date_str = os.path.basename(tiff_path)[10:20]
        print(f"[{i+1}/{n_images}] {date_str}...", end="\r")
        
        try:
            masks, transform, crs = classify_aggressive_shadows(tiff_path)
            
            if masks:
                display_img = np.zeros(masks['glacier'].shape, dtype=np.uint8)
                display_img[masks['glacier'] == 1] = 1 # Grey
                display_img[masks['glacier_lake'] == 1] = 2 # Blue
                
                axes[i].imshow(display_img, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
                axes[i].set_title(date_str, fontsize=10)
                axes[i].axis('off')
        except:
            pass

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    out_path = os.path.join(img_out_dir, "Final_Aggressive_Shadow_Grid.png")
    print(f"\nSaving Aggressive Grid: {out_path}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

if __name__ == "__main__":
    tiff_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    img_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\final_visuals"
    process_aggressive(tiff_dir, img_out)



