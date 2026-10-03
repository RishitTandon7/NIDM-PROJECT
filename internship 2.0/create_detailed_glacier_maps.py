"""
High-Detail Glacier Classification Map Generator
Produces detailed polygonized maps matching the reference style
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.collections import PatchCollection
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from shapely.geometry import shape
import glob
import warnings
warnings.filterwarnings('ignore')

class DetailedGlacierMapper:
    """Creates highly detailed glacier classification maps"""
    
    def __init__(self, output_dir="detailed_glacier_maps"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Exact colors from reference image
        self.colors = {
            'Glacier': '#808080',        # Gray
            'Glacier Lake': '#FF6B6B',   # Red/Pink
            'Non-Glacier': '#4A90E2'     # Blue
        }
        
        self.classifier = RandomForestClassifier(
            n_estimators=150, 
            max_depth=20,
            min_samples_split=5,
            random_state=42, 
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        
    def extract_features(self, vrt_path):
        """Extract spectral features from Sentinel-2 data"""
        print(f"Loading: {os.path.basename(vrt_path)}")
        
        with rasterio.open(vrt_path) as src:
            # Read bands (Sentinel-2 has 10 bands)
            bands = []
            for i in range(1, min(src.count + 1, 11)):
                band = src.read(i).astype(float)
                nodata = src.nodatavals[i-1] if src.nodatavals[i-1] is not None else 0
                band[band == nodata] = 0
                bands.append(band)
            
            bands = np.array(bands)
            height, width = bands[0].shape
            
            # Calculate indices
            print("Calculating spectral indices...")
            
            # NDWI (water detection) - Band3(Green) vs Band8(NIR)
            green = bands[2]  # Band 3
            nir = bands[7] if len(bands) > 7 else bands[4]  # Band 8 or 5
            with np.errstate(divide='ignore', invalid='ignore'):
                ndwi = (green - nir) / (green + nir + 1e-10)
                ndwi = np.nan_to_num(ndwi, 0)
            
            # NDVI (vegetation) - Band4(Red) vs Band8(NIR)
            red = bands[3] if len(bands) > 3 else bands[2]  # Band 4 or 3
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = (nir - red) / (nir + red + 1e-10)
                ndvi = np.nan_to_num(ndvi, 0)
            
            # NDSI (snow/ice) - Band3(Green) vs Band11(SWIR)
            swir = bands[9] if len(bands) > 9 else bands[5]  # Band 11 or 6
            with np.errstate(divide='ignore', invalid='ignore'):
                ndsi = (green - swir) / (green + swir + 1e-10)
                ndsi = np.nan_to_num(ndsi, 0)
            
            # Stack features
            feature_stack = np.dstack([
                *bands[:min(len(bands), 10)],
                ndwi, ndvi, ndsi
            ])
            
            features = feature_stack.reshape(-1, feature_stack.shape[2])
            
            return features, (height, width), src.transform, src.crs
    
    def train_classifier(self, features, sample_size=20000):
        """Train with spectral-based labels"""
        print("Training classifier...")
        
        # Sample for training
        n_pixels = features.shape[0]
        indices = np.random.choice(n_pixels, min(sample_size, n_pixels), replace=False)
        X_train = features[indices]
        
        # Extract indices for classification
        ndwi_idx = -3
        ndvi_idx = -2
        ndsi_idx = -1
        
        ndwi = X_train[:, ndwi_idx]
        ndvi = X_train[:, ndvi_idx]
        ndsi = X_train[:, ndsi_idx]
        
        # Advanced classification logic
        y_train = np.zeros(len(X_train), dtype=int)
        
        # Glacier: High NDSI, low NDVI, minimal water
        glacier_mask = (ndsi > 0.35) & (ndvi < 0.25) & (ndwi < 0.25)
        y_train[glacier_mask] = 0
        
        # Glacier Lake: High NDWI + moderate NDSI (water near ice)
        glacier_lake_mask = (ndwi > 0.25) & (ndsi > 0.15) & (ndsi < 0.45) & (ndvi < 0.3)
        y_train[glacier_lake_mask] = 1
        
        # Non-Glacier: Everything else (rocks, vegetation, etc)
        non_glacier_mask = ~(glacier_mask | glacier_lake_mask)
        y_train[non_glacier_mask] = 2
        
        # Refine classification
        # Very high NDWI = definitely water
        y_train[ndwi > 0.5] = 2  # Non-glacier water
        
        # Very high NVDI = vegetation
        y_train[ndvi > 0.4] = 2
        
        print(f"  Glacier: {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
        print(f"  Glacier Lake: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
        print(f"  Non-Glacier: {np.sum(y_train == 2)} ({np.sum(y_train == 2)/len(y_train)*100:.1f}%)")
        
        # Train
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.classifier.fit(X_train_scaled, y_train)
        
        return self.classifier
    
    def classify_and_map(self, vrt_path):
        """Create detailed classification map"""
        
        # Extract features
        features, (height, width), transform, crs = self.extract_features(vrt_path)
        
        # Train classifier
        self.train_classifier(features)
        
        # Classify all pixels
        print("Classifying image...")
        features_scaled = self.scaler.transform(features)
        predictions = self.classifier.predict(features_scaled)
        classified = predictions.reshape(height, width)
        
        # Polygonize with detail
        print("Creating detailed polygons...")
        mask = classified >= 0
        polygon_shapes = []
        
        for geom, value in shapes(classified.astype(np.int16), mask=mask, transform=transform):
            polygon_shapes.append({
                'geometry': shape(geom),
                'class': int(value)
            })
        
        print(f"Created {len(polygon_shapes)} polygons")
        
        return classified, polygon_shapes, transform, crs
    
    def create_detailed_map(self, vrt_path):
        """Generate final detailed map"""
        
        # Get filename and date
        filename = os.path.basename(vrt_path)
        if 'Sentinel2' in filename:
            date_str = filename.split('_')[1].replace('.vrt', '')
            title_date = f"Sentinel2_{date_str}"
        else:
            title_date = filename.replace('.vrt', '')
        
        print(f"\n{'='*60}")
        print(f"Generating map for: {title_date}")
        print(f"{'='*60}")
        
        # Classify
        classified, polygons, transform, crs = self.classify_and_map(vrt_path)
        
        # Get geographic bounds
        with rasterio.open(vrt_path) as src:
            bounds = src.bounds
            lon_min, lat_min = bounds.left, bounds.bottom
            lon_max, lat_max = bounds.right, bounds.top
        
        # Convert to lat/lon if needed
        if crs and not crs.is_geographic:
            from pyproj import Transformer
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon_min, lat_min = transformer.transform(bounds.left, bounds.bottom)
            lon_max, lat_max = transformer.transform(bounds.right, bounds.top)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot classified raster
        extent = [lon_min, lon_max, lat_min, lat_max]
        
        # Create color map
        cmap = ListedColormap([self.colors['Glacier'], 
                              self.colors['Glacier Lake'], 
                              self.colors['Non-Glacier']])
        
        im = ax.imshow(classified, cmap=cmap, vmin=0, vmax=2, 
                      extent=extent, interpolation='nearest', aspect='auto')
        
        # Add polygon boundaries (black outlines)
        print("Drawing polygon boundaries...")
        from matplotlib.patches import Polygon as MPLPolygon
        from shapely.geometry import mapping
        
        for poly_data in polygons:
            geom = poly_data['geometry']
            
            # Convert to matplotlib polygon
            if geom.geom_type == 'Polygon':
                coords = list(geom.exterior.coords)
                if crs and not crs.is_geographic:
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    coords = [transformer.transform(x, y) for x, y in coords]
                
                # Draw outline
                xs, ys = zip(*coords)
                ax.plot(xs, ys, color='black', linewidth=0.3, alpha=0.7)
            
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    coords = list(poly.exterior.coords)
                    if crs and not crs.is_geographic:
                        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                        coords = [transformer.transform(x, y) for x, y in coords]
                    
                    xs, ys = zip(*coords)
                    ax.plot(xs, ys, color='black', linewidth=0.3, alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Longitude', fontsize=13, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=13, fontweight='bold')
        ax.set_title(f'Polygonized Classified Glacier Map\nLandsat_5 - {title_date}', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Grid
        ax.grid(True, alpha=0.3, color='black', linestyle=':', linewidth=0.5)
        
        # Legend
        legend_patches = [
            mpatches.Patch(color=self.colors['Glacier'], label='Glacier', edgecolor='black'),
            mpatches.Patch(color=self.colors['Glacier Lake'], label='Glacier Lake', edgecolor='black'),
            mpatches.Patch(color=self.colors['Non-Glacier'], label='Non-Glacier', edgecolor='black')
        ]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=11, 
                 framealpha=0.95, edgecolor='black', fancybox=False)
        
        # Border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        
        # Save
        output_path = os.path.join(self.output_dir, f'{title_date}_detailed_map.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=250, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved: {output_path}\n")
        
        plt.show()
        plt.close()
        
        # Print stats
        total_pixels = classified.size
        print(f"Classification Statistics:")
        print(f"  Glacier:       {np.sum(classified == 0):,} pixels ({np.sum(classified == 0)/total_pixels*100:.1f}%)")
        print(f"  Glacier Lake:  {np.sum(classified == 1):,} pixels ({np.sum(classified == 1)/total_pixels*100:.1f}%)")
        print(f"  Non-Glacier:   {np.sum(classified == 2):,} pixels ({np.sum(classified == 2)/total_pixels*100:.1f}%)")


def main():
    """Process Sentinel-2 images"""
    
    print("="*60)
    print("DETAILED GLACIER CLASSIFICATION MAP GENERATOR")
    print("="*60)
    
    mapper = DetailedGlacierMapper()
    
    # Sentinel-2 directory
    vrt_dir = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"
    vrt_files = sorted(glob.glob(os.path.join(vrt_dir, "*.vrt")))
    
    if not vrt_files:
        print("No VRT files found!")
        return
    
    print(f"\nFound {len(vrt_files)} Sentinel-2 images")
    print(f"First: {os.path.basename(vrt_files[0])}")
    print(f"Last:  {os.path.basename(vrt_files[-1])}")
    
    choice = input("\nProcess (f)irst, (l)ast, (a)ll, or (c)ustom number? (f/l/a/c): ").strip().lower()
    
    to_process = []
    if choice == 'f':
        to_process = [vrt_files[0]]
    elif choice == 'l':
        to_process = [vrt_files[-1]]
    elif choice == 'a':
        to_process = vrt_files
    elif choice == 'c':
        try:
            num = int(input("Which file number (1-{})?".format(len(vrt_files))))
            to_process = [vrt_files[num-1]]
        except:
            print("Invalid number, using first file")
            to_process = [vrt_files[0]]
    else:
        to_process = [vrt_files[0]]
    
    # Process
    for vrt_path in to_process:
        try:
            mapper.create_detailed_map(vrt_path)
        except Exception as e:
            print(f"Error processing {os.path.basename(vrt_path)}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"✓ Complete! Maps saved in: {mapper.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()





