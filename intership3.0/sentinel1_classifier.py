
import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import median_filter
from skimage.morphology import disk, closing
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Adjust these paths as needed
INPUT_S1_DIR = r"f:\NIDM-SRM Drone Project Docs\Sentinel1_1"
DEM_PATH = r"f:\NIDM-SRM Drone Project Docs\GLOF project\Data\GLOFDEM.tiff"
OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\sentinel1_classification"

# Thresholds
SLOPE_THRESHOLD_DEG = 8.0  # Strict Flat Terrain to suppress shadows
LAKE_VV_THRESH = -14.0     # Low backscatter for water

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def convert_to_db(linear_data):
    """Convert linear SAR values to dB scale."""
    eps = 1e-10
    return 10 * np.log10(np.maximum(linear_data, eps))

def apply_speckle_filter(data, size=5):
    """Apply Median filter for speckle reduction."""
    return median_filter(data, size=size)

def calculate_slope(dem_array, transform):
    """
    Calculate slope in degrees from a DEM array and its transform.
    Assumes CRS has meters as units.
    """
    res_x = abs(transform[0])
    res_y = abs(transform[4])
    dy, dx = np.gradient(dem_array)
    dy = dy / res_y
    dx = dx / res_x
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad)

def align_dem_to_s1(s1_profile, dem_path):
    """Reproject DEM to match Sentinel-1 grid and calculate slope."""
    with rasterio.open(dem_path) as src_dem:
        dem_reprojected = np.zeros((s1_profile['height'], s1_profile['width']), dtype=np.float32)
        reproject(
            source=rasterio.band(src_dem, 1),
            destination=dem_reprojected,
            src_transform=src_dem.transform,
            src_crs=src_dem.crs,
            dst_transform=s1_profile['transform'],
            dst_crs=s1_profile['crs'],
            resampling=Resampling.bilinear
        )
    return calculate_slope(dem_reprojected, s1_profile['transform'])

def classify_s1_image(tiff_path, dem_path, output_dir):
    filename = os.path.basename(tiff_path)
    date_str = filename.replace('.tif', '').replace('S1_', '')
    print(f"Processing {date_str}...")
    
    # 1. Load S1 Data
    with rasterio.open(tiff_path) as src:
        vv_linear = src.read(1)
        vh_linear = src.read(2)
        profile = src.profile
        
    # 2. Preprocessing (Data already in dB, apply Speckle Filter)
    vv_filt = apply_speckle_filter(vv_linear, size=5)
    
    # 3. Terrain/Slope Alignment
    try:
        slope_map = align_dem_to_s1(profile, dem_path)
    except Exception as e:
        print(f"Error processing DEM: {e}")
        slope_map = np.full_like(vv_filt, 100.0) # Conservative: assume steep if DEM fails
        
    # 4. Classification Logic (Binary: Lake vs Non-Lake)
    # 0 = Non-Lake
    # 1 = Glacial Lake
    
    # "Glacier ice cannot be reliably separated from terrain using Sentinel-1 intensity alone in high-relief regions."
    # Therefore, classifying ONLY lakes.
    
    # Criteria:
    # 1. Low Backscatter: VV < -14 dB
    # 2. Flat Terrain: Slope < 8 deg (Suppress radar shadows which mimic water)
    
    lake_mask = (vv_filt < LAKE_VV_THRESH) & (slope_map < SLOPE_THRESHOLD_DEG)
    
    classified = np.zeros_like(vv_filt, dtype=np.uint8)
    classified[lake_mask] = 1
    
    # 5. Post-processing (Closing on Lake Class ONLY)
    current_lake_mask = (classified == 1)
    closed_lake_mask = closing(current_lake_mask, disk(2)) # 3x3 equivalent approx radius=1 or 2
    
    # update classification: where closed lake is true, set to 1.
    # Note: This might overwrite small glacier pixels or non-glacier pixels inside the lake.
    classified[closed_lake_mask] = 1
    
    # 6. Save GeoTIFF
    os.makedirs(output_dir, exist_ok=True)
    out_name = f"Classified_{date_str}.tif"
    out_path = os.path.join(output_dir, out_name)
    
    profile.update(dtype=rasterio.uint8, count=1, nodata=0) # 0 is a valid class (Non-Glacier), but also BG.
    
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(classified, 1)
        
    # 7. Visualization
    viz_path = str(out_path).replace('.tif', '.png')
    visualize_classification(classified, viz_path, date_str)
    
    return out_path

def visualize_classification(class_map, out_path, title):
    """
    Create a 2-color PNG.
    0: Black (Non-Lake)
    1: Blue (Glacial Lake)
    """
    # Define colormap matching values 0, 1
    colors = ['black', 'blue'] 
    cmap = ListedColormap(colors)
    # Boundaries for 0, 1
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    
    plt.figure(figsize=(10, 10))
    im = plt.imshow(class_map, cmap=cmap, norm=norm, interpolation='nearest')
    plt.title(f"Classification: {title}\n(Black: Non-Lake, Blue: Glacial Lake)")
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', label='Non-Lake'),
        Patch(facecolor='blue', label='Glacial Lake')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.axis('off')
    
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    if not os.path.exists(INPUT_S1_DIR):
        print(f"Error: S1 Directory not found: {INPUT_S1_DIR}")
        return
    if not os.path.exists(DEM_PATH):
        print(f"Error: DEM not found: {DEM_PATH}")
        return

    tiff_files = sorted(glob.glob(os.path.join(INPUT_S1_DIR, "S1_*.tif")))
    print(f"Found {len(tiff_files)} Sentinel-1 images.")
    
    for tiff in tiff_files:
        try:
            classify_s1_image(tiff, DEM_PATH, OUTPUT_DIR)
        except Exception as e:
            print(f"Failed to process {tiff}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
