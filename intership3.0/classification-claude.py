import os
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
import glob

class GlacierClassifier:
    def __init__(self, sentinel_path):
        """
        Initialize the Glacier Classifier
        
        Parameters:
        sentinel_path: Path to Sentinel-2 data directory
        """
        self.sentinel_path = Path(sentinel_path)
        self.classified_images = []
        
    def load_sentinel_bands(self, image_folder):
        """
        Load relevant Sentinel-2 bands for glacier classification
        Typically uses Blue (B2), Green (B3), Red (B4), NIR (B8), SWIR (B11, B12)
        """
        bands = {}
        band_files = {
            'B2': None,  # Blue
            'B3': None,  # Green
            'B4': None,  # Red
            'B8': None,  # NIR
            'B11': None, # SWIR1
            'B12': None  # SWIR2
        }
        
        # Find band files
        for band in band_files.keys():
            pattern = os.path.join(image_folder, f'*{band}*.jp2')
            files = glob.glob(pattern)
            if not files:
                pattern = os.path.join(image_folder, f'*{band}*.tif')
                files = glob.glob(pattern)
            if files:
                band_files[band] = files[0]
        
        # Load bands
        for band_name, file_path in band_files.items():
            if file_path and os.path.exists(file_path):
                with rasterio.open(file_path) as src:
                    bands[band_name] = src.read(1)
                    if band_name == 'B2':
                        self.profile = src.profile
        
        return bands
    
    def calculate_indices(self, bands):
        """
        Calculate spectral indices for glacier classification
        - NDSI (Normalized Difference Snow Index)
        - NDWI (Normalized Difference Water Index)
        - NDVI (Normalized Difference Vegetation Index)
        """
        indices = {}
        
        # Avoid division by zero
        epsilon = 1e-10
        
        # NDSI = (Green - SWIR) / (Green + SWIR)
        if 'B3' in bands and 'B11' in bands:
            green = bands['B3'].astype(float)
            swir = bands['B11'].astype(float)
            indices['NDSI'] = (green - swir) / (green + swir + epsilon)
        
        # NDWI = (Green - NIR) / (Green + NIR)
        if 'B3' in bands and 'B8' in bands:
            green = bands['B3'].astype(float)
            nir = bands['B8'].astype(float)
            indices['NDWI'] = (green - nir) / (green + nir + epsilon)
        
        # NDVI = (NIR - Red) / (NIR + Red)
        if 'B8' in bands and 'B4' in bands:
            nir = bands['B8'].astype(float)
            red = bands['B4'].astype(float)
            indices['NDVI'] = (nir - red) / (nir + red + epsilon)
        
        return indices
    
    def classify_glacier(self, bands, indices):
        """
        Classify pixels into:
        0: Non-glacier
        1: Glacier
        2: Glacier lake
        
        Using rule-based classification with spectral indices
        """
        h, w = bands['B2'].shape
        classified = np.zeros((h, w), dtype=np.uint8)
        
        # Get indices
        ndsi = indices.get('NDSI', np.zeros((h, w)))
        ndwi = indices.get('NDWI', np.zeros((h, w)))
        ndvi = indices.get('NDVI', np.zeros((h, w)))
        
        # Glacier classification (high NDSI, low NDVI)
        glacier_mask = (ndsi > 0.4) & (ndvi < 0.2)
        classified[glacier_mask] = 1
        
        # Glacier lake classification (high NDWI, lower NDSI than glacier)
        lake_mask = (ndwi > 0.3) & (ndsi > 0.1) & (ndsi < 0.4) & (~glacier_mask)
        classified[lake_mask] = 2
        
        return classified
    
    def process_image(self, image_folder, output_path):
        """
        Process a single Sentinel-2 image tile
        """
        print(f"Processing: {image_folder}")
        
        # Load bands
        bands = self.load_sentinel_bands(image_folder)
        
        if not bands:
            print(f"No bands found in {image_folder}")
            return None
        
        # Calculate indices
        indices = self.calculate_indices(bands)
        
        # Classify
        classified = self.classify_glacier(bands, indices)
        
        # Save classified image
        profile = self.profile.copy()
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw'
        )
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(classified, 1)
        
        print(f"Saved classified image to: {output_path}")
        return output_path
    
    def process_all_images(self, output_dir):
        """
        Process all Sentinel-2 images in the directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image folders (typically contain .SAFE or IMG_DATA)
        image_folders = []
        for root, dirs, files in os.walk(self.sentinel_path):
            # Look for folders containing band files
            jp2_files = glob.glob(os.path.join(root, '*B*.jp2'))
            tif_files = glob.glob(os.path.join(root, '*B*.tif'))
            if jp2_files or tif_files:
                image_folders.append(root)
        
        # Remove duplicates
        image_folders = list(set(image_folders))
        
        print(f"Found {len(image_folders)} image folders")
        
        classified_paths = []
        for i, folder in enumerate(image_folders):
            output_path = output_dir / f"classified_{i}.tif"
            result = self.process_image(folder, str(output_path))
            if result:
                classified_paths.append(result)
        
        return classified_paths
    
    def stitch_images(self, classified_paths, output_mosaic):
        """
        Stitch all classified images into a single mosaic
        """
        print(f"\nStitching {len(classified_paths)} images...")
        
        src_files = [rasterio.open(path) for path in classified_paths]
        
        # Merge images
        mosaic, out_trans = merge(src_files)
        
        # Update metadata
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"
        })
        
        # Write mosaic
        with rasterio.open(output_mosaic, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        # Close source files
        for src in src_files:
            src.close()
        
        print(f"Mosaic saved to: {output_mosaic}")
        return output_mosaic
    
    def visualize_result(self, mosaic_path):
        """
        Visualize the final classified mosaic
        """
        with rasterio.open(mosaic_path) as src:
            data = src.read(1)
        
        # Create color map
        colors = ['#8B4513', '#E0E0E0', '#4169E1']  # Brown, Light Gray, Blue
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(data, cmap=cmap, vmin=0, vmax=2)
        plt.colorbar(ticks=[0, 1, 2], label='Class')
        plt.title('Glacier Classification Results')
        
        # Set colorbar labels
        cbar = plt.colorbar(ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(['Non-Glacier', 'Glacier', 'Glacier Lake'])
        
        plt.tight_layout()
        plt.savefig('classification_result.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualization saved as 'classification_result.png'")


def diagnose_directory(sentinel_path):
    """
    Diagnose the directory structure to find Sentinel-2 files
    """
    print("=== DIRECTORY DIAGNOSTIC ===\n")
    print(f"Searching in: {sentinel_path}\n")
    
    # Check if path exists
    if not os.path.exists(sentinel_path):
        print(f"ERROR: Path does not exist: {sentinel_path}")
        return
    
    # Find all files
    all_files = []
    for root, dirs, files in os.walk(sentinel_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    print(f"Total files found: {len(all_files)}\n")
    
    # Group by extension
    extensions = {}
    for file in all_files:
        ext = os.path.splitext(file)[1].lower()
        if ext not in extensions:
            extensions[ext] = []
        extensions[ext].append(file)
    
    print("File types found:")
    for ext, files in sorted(extensions.items()):
        print(f"  {ext}: {len(files)} files")
        if len(files) <= 5:
            for f in files:
                print(f"    - {os.path.basename(f)}")
    
    print("\n")
    
    # Look for Sentinel-2 specific patterns
    sentinel_patterns = ['.jp2', '.tif', '.tiff', 'B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    
    print("Searching for Sentinel-2 band files:")
    for pattern in sentinel_patterns:
        matching = [f for f in all_files if pattern in os.path.basename(f).upper()]
        if matching:
            print(f"\n  Pattern '{pattern}' - {len(matching)} files found:")
            for f in matching[:3]:  # Show first 3
                print(f"    - {f}")
            if len(matching) > 3:
                print(f"    ... and {len(matching) - 3} more")
    
    print("\n" + "="*50 + "\n")
    return all_files


# Main execution
if __name__ == "__main__":
    # Set your paths
    sentinel_data_path = r"F:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    output_directory = r"F:\NIDM-SRM Drone Project Docs\internship 2.0\Classified_Output"
    final_mosaic_path = r"F:\NIDM-SRM Drone Project Docs\internship 2.0\final_glacier_mosaic.tif"
    
    # First, diagnose the directory
    all_files = diagnose_directory(sentinel_data_path)
    
    # Ask user if they want to continue
    print("\nBased on the files found above, please check:")
    print("1. Are there .jp2 or .tif files?")
    print("2. Do the filenames contain band numbers (B02, B03, B04, etc.)?")
    print("3. What is the exact file naming pattern?\n")
    
    response = input("Do you want to continue with classification? (yes/no): ").strip().lower()
    
    if response == 'yes':
        # Initialize classifier
        classifier = GlacierClassifier(sentinel_data_path)
        
        # Process all images
        classified_paths = classifier.process_all_images(output_directory)
        
        if classified_paths:
            # Stitch into final mosaic
            mosaic = classifier.stitch_images(classified_paths, final_mosaic_path)
            
            # Visualize result
            classifier.visualize_result(final_mosaic_path)
            
            print("\n=== Classification Complete ===")
            print(f"Total images processed: {len(classified_paths)}")
            print(f"Final mosaic: {final_mosaic_path}")
        else:
            print("No images were successfully processed. Please check your data directory structure.")
    else:
        print("\nPlease share the output above so I can help fix the file detection logic!")