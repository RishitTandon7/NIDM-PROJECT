"""
Glacier Classification: Glacier, Glacier Lake, Non-Glacier
Uses actual spectral signatures to identify features
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
import glob

class GlacierClassifier:
    """3-class glacier classification"""
    
    def __init__(self, output_dir="final_glacier_maps"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Colors matching reference
        self.colors = ['#808080', '#FF6B6B', '#4A90E2']  # Gray, Red, Blue
        self.class_names = ['Glacier', 'Glacier Lake', 'Non-Glacier']
        
    def load_and_classify(self, vrt_path):
        """Load Sentinel-2 and classify into 3 categories"""
        
        print(f"\nProcessing: {os.path.basename(vrt_path)}")
        
        with rasterio.open(vrt_path) as src:
            # Read all bands
            bands = []
            for i in range(1, min(src.count + 1, 11)):
                band = src.read(i).astype(float)
                nodata = src.nodatavals[i-1] if src.nodatavals[i-1] is not None else 0
                band[band == nodata] = 0
                band[band < 0] = 0  # Remove negative values
                bands.append(band)
            
            height, width = bands[0].shape
            
            # Calculate key indices
            print("Analyzing spectral signatures...")
            
            # NDWI - detects water
            green = bands[2]  # Band 3
            nir = bands[7] if len(bands) > 7 else bands[4]
            ndwi = np.divide(green - nir, green + nir + 1e-10, 
                           out=np.zeros_like(green), where=(green+nir)!=0)
            
            # NDSI - detects ice/snow
            swir = bands[9] if len(bands) > 9 else bands[5]
            ndsi = np.divide(green - swir, green + swir + 1e-10,
                           out=np.zeros_like(green), where=(green+swir)!=0)
            
            # NDVI - detects vegetation
            red = bands[3] if len(bands) > 3 else bands[2]
            ndvi = np.divide(nir - red, nir + red + 1e-10,
                           out=np.zeros_like(nir), where=(nir+red)!=0)
            
            # Brightness (overall reflectance)
            brightness = np.mean(bands[:min(len(bands), 4)], axis=0)
            
            # CLASSIFICATION LOGIC based on spectral signatures
            classified = np.zeros((height, width), dtype=np.uint8)
            
            # Class 0: GLACIER (ice/snow)
            # High NDSI, high brightness, low NDVI
            glacier_mask = (ndsi > 0.4) & (brightness > np.percentile(brightness, 50)) & (ndvi < 0.2)
            classified[glacier_mask] = 0
            
            # Class 1: GLACIER LAKE (water near glaciers)
            # High NDWI, moderate NDSI (transitional), low brightness
            glacier_lake_mask = (
                (ndwi > 0.2) & 
                (ndsi > -0.1) & (ndsi < 0.5) &
                (brightness < np.percentile(brightness, 60)) &
                (ndvi < 0.3)
            )
            classified[glacier_lake_mask] = 1
            
            # Class 2: NON-GLACIER (everything else - rock, vegetation, other water)
            # Everything not classified as glacier or glacier lake
            non_glacier_mask = ~(glacier_mask | glacier_lake_mask)
            classified[non_glacier_mask] = 2
            
            # Additional refinements
            # Very dark areas with high NDWI = definitely water (glacier lake or non-glacier)
            dark_water = (ndwi > 0.3) & (brightness < np.percentile(brightness, 30))
            classified[dark_water] = 1  # Likely glacier lake if dark and wet
            
            # Very bright areas = likely glacier
            very_bright = brightness > np.percentile(brightness, 85)
            classified[very_bright & (ndsi > 0.2)] = 0
            
            # Print stats
            total = height * width
            print(f"\nClassification Results:")
            print(f"  Glacier:       {np.sum(classified==0):>8,} pixels ({np.sum(classified==0)/total*100:>5.1f}%)")
            print(f"  Glacier Lake:  {np.sum(classified==1):>8,} pixels ({np.sum(classified==1)/total*100:>5.1f}%)")
            print(f"  Non-Glacier:   {np.sum(classified==2):>8,} pixels ({np.sum(classified==2)/total*100:>5.1f}%)")
            
            return classified, src.bounds, src.crs, src.transform
    
    def create_map(self, vrt_path):
        """Generate final classification map"""
        
        # Classify
        classified, bounds, crs, transform = self.load_and_classify(vrt_path)
        
        # Get filename
        filename = os.path.basename(vrt_path).replace('.vrt', '')
        
        # Convert bounds to lat/lon if needed
        from pyproj import Transformer
        if crs and not crs.is_geographic:
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon_min, lat_min = transformer.transform(bounds.left, bounds.bottom)
            lon_max, lat_max = transformer.transform(bounds.right, bounds.top)
        else:
            lon_min, lat_min = bounds.left, bounds.bottom
            lon_max, lat_max = bounds.right, bounds.top
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 11))
        
        # Plot
        extent = [lon_min, lon_max, lat_min, lat_max]
        cmap = ListedColormap(self.colors)
        
        im = ax.imshow(classified, cmap=cmap, vmin=0, vmax=2, 
                      extent=extent, interpolation='nearest', aspect='auto')
        
        # Add polygon outlines
        print("Creating polygon boundaries...")
        mask = classified >= 0
        for geom, value in shapes(classified.astype(np.int16), mask=mask, transform=transform):
            from shapely.geometry import shape
            polygon = shape(geom)
            
            # Convert coordinates
            if polygon.geom_type == 'Polygon':
                coords = list(polygon.exterior.coords)
                if crs and not crs.is_geographic:
                    coords = [transformer.transform(x, y) for x, y in coords]
                xs, ys = zip(*coords)
                ax.plot(xs, ys, color='black', linewidth=0.2, alpha=0.6)
        
        # Formatting
        ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')
        ax.set_title(f'Polygonized Classified Glacier Map\nSentinel-2 - {filename}', 
                    fontsize=15, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3, color='black', linestyle=':', linewidth=0.5)
        
        # Legend
        patches = [
            mpatches.Patch(color=self.colors[0], label='Glacier', edgecolor='black'),
            mpatches.Patch(color=self.colors[1], label='Glacier Lake', edgecolor='black'),
            mpatches.Patch(color=self.colors[2], label='Non-Glacier', edgecolor='black')
        ]
        ax.legend(handles=patches, loc='upper right', fontsize=12,
                 framealpha=0.95, edgecolor='black')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        
        # Save
        output_path = os.path.join(self.output_dir, f'{filename}_glacier_map.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nSaved: {output_path}")
        
        plt.show()
        plt.close()


def main():
    print("="*70)
    print("GLACIER CLASSIFICATION: Glacier | Glacier Lake | Non-Glacier")
    print("="*70)
    
    classifier = GlacierClassifier()
    
    # Sentinel-2 data
    vrt_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"
    vrt_files = sorted(glob.glob(os.path.join(vrt_dir, "*.vrt")))
    
    print(f"\nFound {len(vrt_files)} Sentinel-2 scenes")
    
    # Ask which to process
    print(f"  First: {os.path.basename(vrt_files[0])}")
    print(f"  Last:  {os.path.basename(vrt_files[-1])}")
    
    choice = input("\nProcess which? (1) First  (2) Last  (3) Multiple: ").strip()
    
    if choice == '2':
        to_process = [vrt_files[-1]]
    elif choice == '3':
        num = int(input(f"How many to process (1-{len(vrt_files)}): "))
        to_process = vrt_files[:num]
    else:
        to_process = [vrt_files[0]]
    
    # Process
    for vrt_path in to_process:
        try:
            classifier.create_map(vrt_path)
            print("\n" + "="*70)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nCOMPLETE! Maps saved in: {classifier.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()





