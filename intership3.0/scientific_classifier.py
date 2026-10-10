"""
Scientific Glacial Lake Classifier - Based on Published Research
=================================================================
Uses proven thresholds from peer-reviewed literature:
- Huggel et al. (2002) - Glacial lake detection
- Bolch et al. (2011) - Himalayan glacier lakes
- Gardelle et al. (2011) - Mountain lake mapping

Key Strategy:
1. Use ORIGINAL bands (no histogram eq) - HE distorts spectral signatures
2. Apply PROVEN thresholds from literature
3. Conservative approach - only detect OBVIOUS lakes
4. Size filtering - lakes must be > 0.01 km² (100 pixels)

Classes:
- 0: Non-Glacier
- 1: Glacier
- 2: Glacial Lake
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from skimage.measure import label, regionprops
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')


class ScientificGlacierClassifier:
    def __init__(self):
        pass
    
    def classify_scientific(self, blue, green, red, nir, swir, profile):
        """
        Classification using published scientific thresholds
        
        Based on:
        - Huggel et al. (2002): NDWI > 0.4 for water
        - Bolch et al. (2011): NDSI > 0.4 for snow/ice
        - Additional constraints from field validation
        """
        
        print("\nApplying Scientific Classification (Literature-based)...")
        
        # Determine if values are DN or reflectance
        if np.max(blue) > 2.0:
            # DN values - scale to reflectance  
            scale = 10000.0
            print("  Scaling DN values by 10000")
        else:
            scale = 1.0
            print("  Using reflectance values")
        
        # Scale to reflectance (0-1)
        b = np.clip(blue / scale, 0, 1)
        g = np.clip(green / scale, 0, 1)
        r = np.clip(red / scale, 0, 1)
        n = np.clip(nir / scale, 0, 1)
        s = np.clip(swir / scale, 0, 1)
        
        eps = 1e-10
        h, w = blue.shape
        
        # Calculate standard indices
        ndwi = (g - n) / (g + n + eps)
        mndwi = (g - s) / (g + s + eps)
        ndsi = (g - s) / (g + s + eps)
        ndvi = (n - r) / (n + r + eps)
        
        class_map = np.zeros((h, w), dtype=np.uint8)
        
        # ==========================================
        # GLACIER/SNOW (Based on Bolch et al. 2011)
        # ==========================================
        # Published threshold: NDSI > 0.4 AND NIR > threshold
        glacier_mask = (
            (ndsi > 0.4) &                    # Standard published threshold
            (n > 0.1) &                       # Minimum NIR reflectance
            (ndvi < 0.2) &                    # Not vegetation
            ((b + g + r) / 3 > 0.25)          # Reasonable brightness
        )
        
        # ==========================================
        # WATER (Based on Huggel et al. 2002, Gardelle et al. 2011)
        # ==========================================
        # Published thresholds:
        # - NDWI > 0.4 (strict)
        # - MNDWI > 0.3 (Xu 2006)
        # - NIR < 0.15 (water absorbs NIR)
        
        water_mask = (
            (ndwi > 0.4) &                    # Huggel et al. threshold
            (mndwi > 0.3) &                   # Xu 2006 threshold
            (n < 0.15) &                      # Strong NIR absorption
            (s < 0.1) &                       # Strong SWIR absorption
            (ndvi < 0.1) &                    # Not vegetation
            (~glacier_mask)                   # Not glacier
        )
        
        # ==========================================
        # SIZE FILTERING (Based on minimum detectable lake size)
        # ==========================================
        # Gardelle et al. (2011): Minimum lake size 0.01 km²
        # At 10m resolution: 0.01 km² = 100 pixels
        
        MIN_LAKE_SIZE_PIXELS = 100
        MIN_GLACIER_SIZE_PIXELS = 50
        
        print("  Filtering by size (min lake: 100 pixels = 0.01 km²)...")
        
        # Remove small features
        labeled_glacier = label(glacier_mask)
        for region in regionprops(labeled_glacier):
            if region.area < MIN_GLACIER_SIZE_PIXELS:
                glacier_mask[labeled_glacier == region.label] = False
        
        labeled_water = label(water_mask)
        num_lakes = 0
        for region in regionprops(labeled_water):
            if region.area < MIN_LAKE_SIZE_PIXELS:
                water_mask[labeled_water == region.label] = False
            else:
                num_lakes += 1
        
        # Assign classes
        class_map[glacier_mask] = 1
        class_map[water_mask] = 2
        
        # Statistics
        total_px = h * w
        glacier_pct = (np.sum(glacier_mask) / total_px) * 100
        water_pct = (np.sum(water_mask) / total_px) * 100
        terrain_pct = 100 - glacier_pct - water_pct
        
        print(f"  Glacier: {glacier_pct:.2f}%")
        print(f"  Water: {water_pct:.2f}%")
        print(f"  Terrain: {terrain_pct:.2f}%")
        print(f"  Number of lakes detected: {num_lakes}")
        
        # Calculate lake areas
        if num_lakes > 0:
            pixel_area_km2 = (10 * 10) / 1e6  # 10m resolution
            print(f"\n  Lake Statistics:")
            lake_id = 1
            for region in regionprops(label(water_mask)):
                if region.area >= MIN_LAKE_SIZE_PIXELS:
                    area_km2 = region.area * pixel_area_km2
                    print(f"    Lake {lake_id}: {area_km2:.4f} km²")
                    lake_id += 1
        
        return class_map, {
            'ndwi': ndwi,
            'mndwi': mndwi,
            'ndsi': ndsi,
            'ndvi': ndvi,
            'num_lakes': num_lakes
        }, (b, g, r, n, s)
    
    def create_visualization(self, class_map, indices, bands, date_str, output_path):
        """Create clean visualization"""
        
        b, g, r, n, s = bands
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. True Color RGB
        rgb = np.dstack([r, g, b])
        rgb_display = np.clip(rgb * 3, 0, 1)  # Brighten for display
        axes[0, 0].imshow(rgb_display)
        axes[0, 0].set_title('True Color RGB', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. False Color (NIR-R-G)
        false_color = np.dstack([n, r, g])
        false_color_display = np.clip(false_color * 3, 0, 1)
        axes[0, 1].imshow(false_color_display)
        axes[0, 1].set_title('False Color (NIR-R-G)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 3. Classification
        colors = ['#8B7355', '#FFFFFF', '#0080FF']
        cmap = ListedColormap(colors)
        axes[0, 2].imshow(class_map, cmap=cmap, vmin=0, vmax=2)
        axes[0, 2].set_title(f'Classification\n({indices["num_lakes"]} lakes detected)', 
                            fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        legend = [
            Patch(facecolor='#8B7355', label='Terrain'),
            Patch(facecolor='#FFFFFF', label='Glacier/Snow'),
            Patch(facecolor='#0080FF', label='Lake')
        ]
        axes[0, 2].legend(handles=legend, loc='lower right', fontsize=9)
        
        # 4. NDWI
        im1 = axes[1, 0].imshow(indices['ndwi'], cmap='RdYlBu', vmin=-0.5, vmax=0.8)
        axes[1, 0].set_title('NDWI (Water Index)', fontsize=11)
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
        
        # 5. NDSI
        im2 = axes[1, 1].imshow(indices['ndsi'], cmap='RdYlGn', vmin=-0.5, vmax=0.8)
        axes[1, 1].set_title('NDSI (Snow Index)', fontsize=11)
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
        
        # 6. Overlay
        overlay = rgb_display.copy()
        overlay[class_map == 1] = [0.8, 1.0, 1.0]  # Cyan for glacier
        overlay[class_map == 2] = [0, 0.5, 1.0]    # Blue for lake
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Overlay Check', fontsize=11)
        axes[1, 2].axis('off')
        
        fig.suptitle(f'Scientific Glacial Lake Classification - {date_str}\n' +
                    'Based on Huggel (2002), Bolch (2011), Gardelle (2011)',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")


def test_scientific(tiff_path, output_dir):
    """Test scientific classifier"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    date_str = os.path.basename(tiff_path)[10:20]
    if len(date_str) < 5:
        date_str = "test"
    
    print(f"\n{'='*70}")
    print(f"SCIENTIFIC GLACIAL LAKE CLASSIFIER")
    print(f"{'='*70}")
    print(f"Image: {os.path.basename(tiff_path)}")
    print(f"Date: {date_str}")
    print(f"{'='*70}")
    
    # Load data
    with rasterio.open(tiff_path) as src:
        profile = src.profile
        blue = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        red = src.read(3).astype(np.float32)
        nir = src.read(4).astype(np.float32)
        
        if src.count >= 9:
            swir = src.read(9).astype(np.float32)
        elif src.count >= 5:
            swir = src.read(5).astype(np.float32)
        else:
            swir = np.zeros_like(blue)
    
    # Classify
    classifier = ScientificGlacierClassifier()
    class_map, indices, bands = classifier.classify_scientific(blue, green, red, nir, swir, profile)
    
    # Save visualization
    viz_path = os.path.join(output_dir, f"Scientific_{date_str}.png")
    classifier.create_visualization(class_map, indices, bands, date_str, viz_path)
    
    # Save GeoTIFF
    tif_path = os.path.join(output_dir, f"Classification_{date_str}.tif")
    prof_out = profile.copy()
    prof_out.update({'dtype': rasterio.uint8, 'count': 1, 'compress': 'lzw'})
    with rasterio.open(tif_path, 'w', **prof_out) as dst:
        dst.write(class_map, 1)
    print(f"✓ Saved: {tif_path}")
    
    # Save vectors
    for class_val, class_name in [(1, 'Glacier'), (2, 'Lake')]:
        mask = (class_map == class_val).astype(np.uint8)
        if np.any(mask):
            results = [
                {'properties': {'class': class_name, 'date': date_str}, 'geometry': s}
                for s, v in shapes(mask, mask=mask.astype(bool), transform=profile['transform'])
            ]
            if results:
                gdf = gpd.GeoDataFrame.from_features(results, crs=profile['crs'])
                vec_path = os.path.join(output_dir, f"{class_name}_{date_str}.geojson")
                gdf.to_file(vec_path, driver='GeoJSON')
                print(f"✓ Saved: {vec_path}")
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}\n")
    
    return viz_path


if __name__ == "__main__":
    TIFF_DIR = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    OUTPUT_DIR = r"f:\NIDM-SRM Drone Project Docs\intership3.0\scientific_test"
    
    import glob
    tiff_files = sorted(glob.glob(os.path.join(TIFF_DIR, "*.tif")))
    
    if tiff_files:
        result = test_scientific(tiff_files[0], OUTPUT_DIR)
        print(f"Check result: {result}")
    else:
        print("No TIFF files found!")
