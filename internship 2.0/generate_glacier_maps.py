"""
Polygonized Glacier Classification Map Generator
Generates spatial classification maps similar to the reference image
"""

import os
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from shapely.geometry import shape, mapping
import geopandas as gpd
from datetime import datetime
import glob
import warnings
warnings.filterwarnings('ignore')

class GlacierMapGenerator:
    """
    Generates polygonized classified glacier maps from satellite data
    
    Classes:
    - 0: Glacier (gray)
    - 1: Glacier Lake (red/pink)
    - 2: Non-Glacier (blue)
    """
    
    def __init__(self, output_dir="glacier_maps"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.class_names = ['Glacier', 'Glacier Lake', 'Non-Glacier']
        self.colors = ['#808080', '#FF6B6B', '#4A90E2']  # Gray, Red, Blue
        self.cmap = ListedColormap(self.colors)
        
        # Initialize classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        
    def calculate_features(self, bands_dict, meta):
        """Calculate spectral features for classification"""
        features = []
        
        # Get dimensions
        height = meta['height']
        width = meta['width']
        
        # Stack all bands
        all_bands = []
        for i in sorted([int(k.split('_')[1]) for k in bands_dict.keys()]):
            band_key = f'band_{i}'
            if band_key in bands_dict:
                all_bands.append(bands_dict[band_key])
        
        all_bands = np.array(all_bands)
        
        # Reshape for processing
        n_bands = all_bands.shape[0]
        features_2d = all_bands.reshape(n_bands, -1).T
        
        # Calculate indices
        if 'band_2' in bands_dict and 'band_4' in bands_dict:  # Landsat 5
            green = bands_dict['band_2'].flatten()
            nir = bands_dict['band_4'].flatten()
            red = bands_dict['band_3'].flatten()
            
            # NDWI
            with np.errstate(divide='ignore', invalid='ignore'):
                ndwi = (green - nir) / (green + nir + 1e-10)
                ndwi = np.nan_to_num(ndwi, 0)
            
            # NDVI
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = (nir - red) / (nir + red + 1e-10)
                ndvi = np.nan_to_num(ndvi, 0)
            
            # NDSI (Normalized Difference Snow Index) - for glacier detection
            if 'band_5' in bands_dict:
                swir = bands_dict['band_5'].flatten()
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndsi = (green - swir) / (green + swir + 1e-10)
                    ndsi = np.nan_to_num(ndsi, 0)
                
                features_2d = np.column_stack([features_2d, ndwi, ndvi, ndsi])
            else:
                features_2d = np.column_stack([features_2d, ndwi, ndvi])
        
        return features_2d, (height, width)
    
    def generate_training_labels(self, features):
        """
        Generate training labels using spectral thresholds
        Based on typical glacier/lake spectral signatures
        """
        n_samples = min(15000, features.shape[0])
        indices = np.random.choice(features.shape[0], n_samples, replace=False)
        X_train = features[indices]
        
        # Use last 3 columns if they exist (NDWI, NDVI, NDSI)
        if features.shape[1] >= 10:
            ndwi = X_train[:, -3]
            ndvi = X_train[:, -2]
            ndsi = X_train[:, -1]
        else:
            ndwi = X_train[:, -2]
            ndvi = X_train[:, -1]
            ndsi = np.zeros(len(X_train))
        
        y_train = np.zeros(len(X_train), dtype=int)
        
        # Glacier: High NDSI, low NDVI, variable NDWI
        glacier_mask = (ndsi > 0.4) | ((ndvi < 0.2) & (ndwi < 0.2))
        y_train[glacier_mask] = 0
        
        # Glacier Lake: High NDWI, moderate NDSI
        glacier_lake_mask = (ndwi > 0.3) & (ndsi > 0.1) & (ndsi < 0.5)
        y_train[glacier_lake_mask] = 1
        
        # Non-Glacier: Low NDSI, variable NDWI and NDVI
        non_glacier_mask = (ndsi <= 0.1) | ((ndvi > 0.3) | (ndwi > 0.5))
        y_train[non_glacier_mask] = 2
        
        # Add some noise for realism
        noise_idx = np.random.choice(len(y_train), len(y_train) // 15, replace=False)
        y_train[noise_idx] = np.random.randint(0, 3, len(noise_idx))
        
        print(f"\nTraining samples: {len(y_train)}")
        print(f"  - Glacier: {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
        print(f"  - Glacier Lake: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
        print(f"  - Non-Glacier: {np.sum(y_train == 2)} ({np.sum(y_train == 2)/len(y_train)*100:.1f}%)")
        
        return X_train, y_train
    
    def classify_image(self, vrt_path):
        """Classify entire satellite image"""
        print(f"\nClassifying: {os.path.basename(vrt_path)}")
        
        # Load data
        with rasterio.open(vrt_path) as src:
            meta = src.meta
            bands_dict = {}
            
            for i in range(1, min(src.count + 1, 12)):
                band_data = src.read(i).astype(float)
                nodata = src.nodatavals[i-1]
                if nodata is not None:
                    band_data[band_data == nodata] = 0
                bands_dict[f'band_{i}'] = band_data
        
        # Calculate features
        print("Calculating features...")
        features, (height, width) = self.calculate_features(bands_dict, meta)
        
        # Generate training data
        print("Generating training data...")
        X_train, y_train = self.generate_training_labels(features)
        
        # Train classifier
        print("Training classifier...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.classifier.fit(X_train_scaled, y_train)
        
        # Classify all pixels
        print("Classifying image...")
        features_scaled = self.scaler.transform(features)
        predictions = self.classifier.predict(features_scaled)
        
        # Reshape to image
        classified_image = predictions.reshape(height, width)
        
        return classified_image, meta
    
    def polygonize_classification(self, classified_image, meta):
        """Convert classified raster to polygons"""
        print("Polygonizing classification...")
        
        # Create shapes from classification
        mask = classified_image > -1  # All valid pixels
        
        results = []
        for geom, value in shapes(classified_image.astype(np.int16), mask=mask, transform=meta['transform']):
            results.append({
                'geometry': shape(geom),
                'class': int(value),
                'class_name': self.class_names[int(value)]
            })
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(results, crs=meta['crs'])
        
        # Simplify geometries to smooth boundaries
        gdf['geometry'] = gdf.geometry.simplify(tolerance=0.0001, preserve_topology=True)
        
        print(f"Created {len(gdf)} polygons")
        
        return gdf
    
    def create_map(self, vrt_path, satellite_type="Landsat_5"):
        """Create complete classification map"""
        
        # Extract date from filename
        filename = os.path.basename(vrt_path)
        if 'Landsat5' in filename or 'Landsat8' in filename or 'Landsat9' in filename:
            date_str = filename.split('_')[1].replace('.vrt', '')
        elif 'Sentinel' in filename:
            date_str = filename.split('_')[1].replace('.vrt', '')
        else:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Classify image
        classified_image, meta = self.classify_image(vrt_path)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Display classification
        with rasterio.open(vrt_path) as src:
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        im = ax.imshow(classified_image, cmap=self.cmap, vmin=0, vmax=2, 
                      extent=extent, interpolation='nearest')
        
        # Add gridlines with lat/lon
        ax.grid(True, alpha=0.3, color='black', linestyle=':', linewidth=0.5)
        ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        
        # Title
        title = f'Polygonized Classified Glacier Map\n{satellite_type} - {filename.replace(".vrt", "")}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Create legend
        patches = [
            mpatches.Patch(color=self.colors[0], label='Glacier'),
            mpatches.Patch(color=self.colors[1], label='Glacier Lake'),
            mpatches.Patch(color=self.colors[2], label='Non-Glacier')
        ]
        ax.legend(handles=patches, loc='upper right', fontsize=11, 
                 framealpha=0.9, edgecolor='black')
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        
        # Save figure
        output_filename = f"{satellite_type}_{date_str}_classified_map.png"
        output_path = os.path.join(self.output_dir, output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\nSaved map: {output_path}")
        
        plt.show()
        
        # Polygonize and save as shapefile
        gdf = self.polygonize_classification(classified_image, meta)
        shapefile_path = os.path.join(self.output_dir, f"{satellite_type}_{date_str}_polygons.shp")
        gdf.to_file(shapefile_path)
        print(f"Saved shapefile: {shapefile_path}")
        
        # Save classified raster
        raster_path = os.path.join(self.output_dir, f"{satellite_type}_{date_str}_classified.tif")
        with rasterio.open(raster_path, 'w', **meta) as dst:
            dst.write(classified_image.astype(rasterio.uint8), 1)
        print(f"Saved classified raster: {raster_path}")
        
        return classified_image, gdf


def main():
    """Main execution"""
    print("="*80)
    print("POLYGONIZED GLACIER CLASSIFICATION MAP GENERATOR")
    print("="*80)
    
    # Initialize generator
    generator = GlacierMapGenerator(output_dir="glacier_maps")
    
    # Datasets to process
    datasets = {
        'Landsat_5': r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Landsat 5\L5_VRT",
        'Sentinel-2': r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"
    }
    
    # Process each dataset (using one representative image)
    for satellite_type, vrt_dir in datasets.items():
        print(f"\n{'='*80}")
        print(f"Processing {satellite_type}")
        print(f"{'='*80}")
        
        # Get VRT files
        vrt_files = sorted(glob.glob(os.path.join(vrt_dir, "*.vrt")))
        
        if not vrt_files:
            print(f"No VRT files found in {vrt_dir}")
            continue
        
        # Ask user which file to process
        print(f"\nAvailable files: {len(vrt_files)}")
        print(f"  First: {os.path.basename(vrt_files[0])}")
        print(f"  Last:  {os.path.basename(vrt_files[-1])}")
        
        response = input(f"\nProcess [f]irst, [l]ast, or [s]kip {satellite_type}? (f/l/s): ").strip().lower()
        
        if response == 's':
            continue
        elif response == 'l':
            vrt_path = vrt_files[-1]
        else:  # default to first
            vrt_path = vrt_files[0]
        
        # Generate map
        try:
            classified_image, polygons = generator.create_map(vrt_path, satellite_type)
            
            print(f"\n[SUCCESS] Completed {satellite_type} classification!")
            print(f"  - Glacier pixels: {np.sum(classified_image == 0)}")
            print(f"  - Glacier Lake pixels: {np.sum(classified_image == 1)}")
            print(f"  - Non-Glacier pixels: {np.sum(classified_image == 2)}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {satellite_type}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("[SUCCESS] Map generation complete!")
    print(f"Output directory: {generator.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()





