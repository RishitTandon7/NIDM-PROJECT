"""
Adaptive Glacier Classifier using K-Means Clustering
Method: Unsupervised Machine Learning (K-Means) to adaptively find Snow/Shadow/Water 
thresholds for EACH image individually.
Solves accuracy issues caused by varying lighting conditions.
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
from sklearn.cluster import MiniBatchKMeans

def classify_adaptive_kmeans(vrt_path):
    with rasterio.open(vrt_path) as src:
        # Read Bands 2,3,4,8 (Blue, Green, Red, NIR)
        try:
            if src.count == 4:
                blue  = src.read(1).astype(float)
                green = src.read(2).astype(float)
                red   = src.read(3).astype(float)
                nir   = src.read(4).astype(float)
            else:
                blue  = src.read(2).astype(float)
                green = src.read(3).astype(float)
                red   = src.read(4).astype(float)
                nir   = src.read(8).astype(float)
        except:
            # Fallback
            blue  = src.read(1).astype(float)
            green = src.read(2).astype(float)
            red   = src.read(3).astype(float)
            nir   = src.read(4).astype(float)
            
        transform = src.transform
        crs = src.crs

    # --- FEATURE ENGINEERING ---
    np.seterr(divide='ignore', invalid='ignore')
    
    # Normalize to 0-1 range for ML
    def norm(band):
        p2, p98 = np.percentile(band[band>0], [2, 98])
        return np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)

    n_blue = norm(blue)
    n_green = norm(green)
    n_red = norm(red)
    n_nir = norm(nir)
    
    # Indices are critical features
    ndwi = (green - nir) / (green + nir + 1e-6)
    ndvi = (nir - red) / (nir + red + 1e-6)
    brightness = (n_blue + n_green + n_red) / 3.0
    
    # Stack features for clustering
    # Shape: (N_pixels, N_features)
    mask_valid = (blue > 0) # Ignore NoData
    
    features = np.vstack([
        n_blue[mask_valid],
        n_green[mask_valid],
        n_red[mask_valid],
        n_nir[mask_valid],
        ndwi[mask_valid],
        brightness[mask_valid]
    ]).T

    # --- K-MEANS CLUSTERING ---
    # We ask for 5 Clusters:
    # 1. Deep Shadow/Water
    # 2. Dark Rock
    # 3. Light Rock
    # 4. Dirty Ice
    # 5. Bright Snow
    
    n_clusters = 5
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3, batch_size=2000)
    clusters = kmeans.fit_predict(features)
    
    # Assign clusters back to image
    cluster_img = np.full(blue.shape, -1, dtype=int)
    cluster_img[mask_valid] = clusters
    
    # --- ANALYZE CLUSTERS ---
    # We need to identifying which cluster is which.
    # Calculate mean Brightness and NDWI for each cluster center.
    
    cluster_props = []
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        # Features: B, G, R, NIR, NDWI, Brightness
        # Index: 0, 1, 2, 3, 4, 5
        mean_brightness = center[5]
        mean_ndwi = center[4]
        mean_ndvi = (center[3] - center[2]) / (center[3] + center[2] + 1e-6)
        
        cluster_props.append({
            'id': i,
            'bright': mean_brightness,
            'ndwi': mean_ndwi,
            'ndvi': mean_ndvi
        })
        
    # Sort by Brightness
    sorted_clusters = sorted(cluster_props, key=lambda x: x['bright'])
    
    # Mapping Logic:
    # Lowest Brightness (0 and maybe 1) -> Shadow or Water
    # Highest Brightness (4 and check 3) -> Snow/Glacier
    # Middle -> Non-Glacier Rock
    
    # Init Masks
    glacier_mask = np.zeros_like(blue, dtype=bool)
    lake_mask = np.zeros_like(blue, dtype=bool)
    nonglacier_mask = np.ones_like(blue, dtype=bool) # Default
    
    # 1. GLACIER IDENTIFICATION
    # The brightest cluster is definitely snow.
    brightest_id = sorted_clusters[-1]['id']
    glacier_mask[cluster_img == brightest_id] = True
    
    # Check 2nd brightest (Dirty Ice?)
    second_bright = sorted_clusters[-2]
    # If it's reasonably bright (> 0.4) and low NDVI, it's probably dirty ice
    if second_bright['bright'] > 0.4 and second_bright['ndvi'] < 0.2:
        glacier_mask[cluster_img == second_bright['id']] = True
        
    # 2. LAKE vs SHADOW IDENTIFICATION
    # The darkest clusters are Shadows OR Water.
    # Distinguish using NDWI. Water has High NDWI (>0.2). Shadow has low/moderate.
    
    darkest_ids = [c['id'] for c in sorted_clusters[:2]] # Check bottom 2
    
    for cid in darkest_ids:
        # Get pixels for this cluster
        c_mask = (cluster_img == cid)
        
        # Split this cluster internally:
        # Pixels with High NDWI -> Lake
        # Pixels with Low NDWI -> Shadow (Non-Glacier)
        
        lake_pixels = c_mask & (ndwi > 0.15) & (n_nir < 0.15) # Refine check
        lake_mask[lake_pixels] = True
        
        # The rest of this cluster remains Non-Glacier (Shadow)
    
    # Cleanup
    # Glacier cannot be Lake
    glacier_mask = glacier_mask & (~lake_mask)
    nonglacier_mask = (~glacier_mask) & (~lake_mask)
    
    # Filter
    glacier_clean = median_filter(glacier_mask.astype(np.uint8), size=3)
    lake_clean = median_filter(lake_mask.astype(np.uint8), size=3)
    nonglacier_clean = median_filter(nonglacier_mask.astype(np.uint8), size=3)
    
    return {
        "glacier": glacier_clean,
        "glacier_lake": lake_clean,
        "nonglacier": nonglacier_clean
    }, transform, crs

def process_adaptive(vrt_dir, json_out_dir, img_out_dir):
    os.makedirs(json_out_dir, exist_ok=True)
    os.makedirs(img_out_dir, exist_ok=True)
    
    vrt_files = sorted(glob.glob(os.path.join(vrt_dir, "*.vrt")))
    print(f"Running Adaptive ML Classification on {len(vrt_files)} scenes...")
    
    # Setup Grid
    n_images = len(vrt_files)
    cols = 6
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
    fig.suptitle('Adaptive AI Classification (K-Means)\nGrey: Glacier | Blue: Lake | Brown: Non-Glacier', fontsize=24, y=0.99)
    axes = axes.flatten()
    cmap = ListedColormap(['#8B4513', '#D3D3D3', '#00BFFF']) # Light Grey for Snow

    for i, vrt_path in enumerate(vrt_files):
        date_str = os.path.basename(vrt_path)[10:20]
        print(f"[{i+1}/{n_images}] {date_str}...", end="\r")
        
        try:
            masks, transform, crs = classify_adaptive_kmeans(vrt_path)
            
            # Vectors
            for class_name, mask_arr in masks.items():
                geoms = []
                if np.sum(mask_arr) > 0:
                     for geom, val in shapes(mask_arr, mask=(mask_arr==1), transform=transform):
                        geoms.append(shape(geom))
                if geoms:
                    gdf = gpd.GeoDataFrame({'geometry': geoms, 'date': date_str, 'class': class_name}, crs=crs)
                    gdf.to_file(os.path.join(json_out_dir, f"{class_name}_{date_str}.geojson"), driver='GeoJSON')
            
            # Display
            display_img = np.zeros(masks['glacier'].shape, dtype=np.uint8) 
            display_img[masks['glacier'] == 1] = 1 
            display_img[masks['glacier_lake'] == 1] = 2 
            
            axes[i].imshow(display_img, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
            axes[i].set_title(date_str, fontsize=10)
            axes[i].axis('off')
            
        except Exception as e:
            print(f"Error {date_str}: {e}")

    # Clear empty
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    out_path = os.path.join(img_out_dir, "Final_Adaptive_ML_Grid.png")
    print(f"\nSaving Adaptive Grid: {out_path}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

if __name__ == "__main__":
    vrt_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"
    json_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\adaptive_vectors_v2"
    img_out = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\final_visuals"
    process_adaptive(vrt_dir, json_out, img_out)



