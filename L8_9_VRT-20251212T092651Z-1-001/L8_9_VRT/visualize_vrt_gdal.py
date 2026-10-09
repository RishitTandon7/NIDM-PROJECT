"""
Script to visualize VRT files using GDAL (osgeo)
For Landsat 8/9 data
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from osgeo import gdal
    gdal.UseExceptions()
    GDAL_AVAILABLE = True
except ImportError:
    print("WARNING: GDAL not available, trying rasterio as fallback...")
    import rasterio
    GDAL_AVAILABLE = False


class VRTVisualizerGDAL:
    def __init__(self, vrt_directory):
        self.vrt_directory = vrt_directory
        self.vrt_files = sorted(glob.glob(os.path.join(vrt_directory, "*.vrt")))
        print(f"\nFound {len(self.vrt_files)} VRT files in {vrt_directory}")
        
    def print_vrt_info_gdal(self, vrt_path):
        """Print VRT info using GDAL"""
        print(f"\n{'='*80}")
        print(f"VRT File: {os.path.basename(vrt_path)}")
        print(f"{'='*80}")
        
        try:
            dataset = gdal.Open(vrt_path, gdal.GA_ReadOnly)
            if dataset is None:
                print(f"ERROR: Could not open {vrt_path}")
                return None
            
            print(f"\n[BASIC INFORMATION]")
            print(f"  Driver: {dataset.GetDriver().ShortName}/{dataset.GetDriver().LongName}")
            print(f"  Size: {dataset.RasterXSize} x {dataset.RasterYSize} pixels")
            print(f"  Number of Bands: {dataset.RasterCount}")
            
            print(f"\n[GEOSPATIAL INFORMATION]")
            projection = dataset.GetProjection()
            if projection:
                print(f"  Projection: {projection[:150]}...")
            
            geotransform = dataset.GetGeoTransform()
            if geotransform:
                print(f"  GeoTransform:")
                print(f"    Origin: ({geotransform[0]:.2f}, {geotransform[3]:.2f})")
                print(f"    Pixel Size: ({geotransform[1]:.2f}, {geotransform[5]:.2f})")
            
            print(f"\n[BAND INFORMATION]")
            for i in range(1, min(dataset.RasterCount + 1, 12)):
                band = dataset.GetRasterBand(i)
                print(f"  Band {i}: {gdal.GetDataTypeName(band.DataType)}, NoData={band.GetNoDataValue()}")
                
                try:
                    stats = band.ComputeStatistics(False)
                    if stats:
                        print(f"    Stats: Min={stats[0]:.2f}, Max={stats[1]:.2f}, Mean={stats[2]:.2f}, StdDev={stats[3]:.2f}")
                except:
                    pass
            
            dataset = None  # Close dataset
            return True
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return False
    
    def print_vrt_info_rasterio(self, vrt_path):
        """Print VRT info using Rasterio (fallback)"""
        print(f"\n{'='*80}")
        print(f"VRT File: {os.path.basename(vrt_path)}")
        print(f"{'='*80}")
        
        try:
            with rasterio.open(vrt_path) as src:
                print(f"\n[BASIC INFORMATION]")
                print(f"  Driver: {src.driver}")
                print(f"  Size: {src.width} x {src.height} pixels")
                print(f"  Number of Bands: {src.count}")
                print(f"  Data Type: {src.dtypes[0]}")
                
                print(f"\n[GEOSPATIAL INFORMATION]")
                print(f"  CRS: {src.crs}")
                print(f"  Bounds: {src.bounds}")
                print(f"  Resolution: {src.res[0]:.2f} x {src.res[1]:.2f}")
                
                print(f"\n[BAND INFORMATION]")
                for i in range(1, min(src.count + 1, 12)):
                    band_data = src.read(i)
                    nodata = src.nodatavals[i-1]
                    
                    if nodata is not None:
                        valid_data = band_data[band_data != nodata]
                    else:
                        valid_data = band_data
                    
                    if len(valid_data) > 0:
                        print(f"  Band {i}: Min={np.min(valid_data):.2f}, Max={np.max(valid_data):.2f}, Mean={np.mean(valid_data):.2f}")
                
                return True
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return False
    
    def read_band_gdal(self, vrt_path, band_number):
        """Read band using GDAL"""
        dataset = gdal.Open(vrt_path, gdal.GA_ReadOnly)
        if dataset is None:
            return None, None
        
        band = dataset.GetRasterBand(band_number)
        data = band.ReadAsArray().astype(float)
        nodata = band.GetNoDataValue()
        
        if nodata is not None:
            data[data == nodata] = np.nan
        
        dataset = None
        return data, nodata
    
    def read_band_rasterio(self, vrt_path, band_number):
        """Read band using Rasterio (fallback)"""
        with rasterio.open(vrt_path) as src:
            data = src.read(band_number).astype(float)
            nodata = src.nodatavals[band_number-1]
            
            if nodata is not None:
                data[data == nodata] = np.nan
            
            return data, nodata
    
    def visualize_single_vrt(self, vrt_path, save_figure=True):
        """Create comprehensive visualization for a single VRT file"""
        try:
            filename = os.path.basename(vrt_path).replace('.vrt', '')
            print(f"\nVisualizing: {filename}")
            
            # Create figure
            fig = plt.figure(figsize=(24, 14))
            gs = GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.3)
            
            fig.suptitle(f'Landsat 8/9 Data Visualization: {filename}', 
                       fontsize=18, fontweight='bold')
            
            # Read bands
            bands_data = []
            for i in range(1, 12):  # Landsat 8/9 has 11 bands
                if GDAL_AVAILABLE:
                    data, _ = self.read_band_gdal(vrt_path, i)
                else:
                    data, _ = self.read_band_rasterio(vrt_path, i)
                
                if data is not None:
                    bands_data.append(data)
                else:
                    break
            
            print(f"  Loaded {len(bands_data)} bands")
            
            # Band names for Landsat 8/9
            band_names = [
                'Band 1 (Coastal/Aerosol)',
                'Band 2 (Blue)',
                'Band 3 (Green)',
                'Band 4 (Red)',
                'Band 5 (NIR)',
                'Band 6 (SWIR1)',
                'Band 7 (SWIR2)',
                'Band 8 (Pan)',
                'Band 9 (Cirrus)',
                'Band 10 (TIRS1)',
                'Band 11 (TIRS2)'
            ]
            
            # Plot first 10 bands
            for idx in range(min(10, len(bands_data))):
                row = idx // 5
                col = idx % 5
                ax = fig.add_subplot(gs[row, col])
                
                im = ax.imshow(bands_data[idx], cmap='gray')
                ax.set_title(band_names[idx], fontsize=9)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Natural Color RGB (Bands 4, 3, 2)
            if len(bands_data) >= 4:
                ax_rgb = fig.add_subplot(gs[2, 0])
                rgb = self._create_rgb_composite(bands_data[3], bands_data[2], bands_data[1])
                ax_rgb.imshow(rgb)
                ax_rgb.set_title('Natural Color (4,3,2)', fontsize=10, fontweight='bold')
                ax_rgb.axis('off')
            
            # False Color - Vegetation (Bands 5, 4, 3)
            if len(bands_data) >= 5:
                ax_false = fig.add_subplot(gs[2, 1])
                false_color = self._create_rgb_composite(bands_data[4], bands_data[3], bands_data[2])
                ax_false.imshow(false_color)
                ax_false.set_title('False Color (5,4,3)', fontsize=10, fontweight='bold')
                ax_false.axis('off')
            
            # SWIR Composite (Bands 7, 6, 4)
            if len(bands_data) >= 7:
                ax_swir = fig.add_subplot(gs[2, 2])
                swir = self._create_rgb_composite(bands_data[6], bands_data[5], bands_data[3])
                ax_swir.imshow(swir)
                ax_swir.set_title('SWIR (7,6,4)', fontsize=10, fontweight='bold')
                ax_swir.axis('off')
            
            # NDVI (Normalized Difference Vegetation Index)
            if len(bands_data) >= 5:
                ax_ndvi = fig.add_subplot(gs[2, 3])
                ndvi = self._calculate_ndvi(bands_data[4], bands_data[3])
                im_ndvi = ax_ndvi.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
                ax_ndvi.set_title('NDVI', fontsize=10, fontweight='bold')
                ax_ndvi.axis('off')
                plt.colorbar(im_ndvi, ax=ax_ndvi, fraction=0.046, pad=0.04)
            
            # NDWI (Normalized Difference Water Index)
            if len(bands_data) >= 6:
                ax_ndwi = fig.add_subplot(gs[2, 4])
                ndwi = self._calculate_ndwi(bands_data[2], bands_data[5])  # Green, SWIR1
                im_ndwi = ax_ndwi.imshow(ndwi, cmap='Blues', vmin=-1, vmax=1)
                ax_ndwi.set_title('NDWI (Water)', fontsize=10, fontweight='bold')
                ax_ndwi.axis('off')
                plt.colorbar(im_ndwi, ax=ax_ndwi, fraction=0.046, pad=0.04)
            
            if save_figure:
                output_dir = os.path.dirname(vrt_path)
                output_path = os.path.join(output_dir, f'{filename}_visualization.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"  Saved to: {output_path}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"ERROR visualizing {vrt_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _create_rgb_composite(self, red, green, blue):
        """Create normalized RGB composite"""
        def normalize(band):
            valid_data = band[~np.isnan(band)]
            if len(valid_data) > 0:
                p2, p98 = np.percentile(valid_data, [2, 98])
                normalized = (band - p2) / (p98 - p2)
                return np.clip(normalized, 0, 1)
            return band
        
        r_norm = normalize(red)
        g_norm = normalize(green)
        b_norm = normalize(blue)
        
        return np.dstack((r_norm, g_norm, b_norm))
    
    def _calculate_ndvi(self, nir, red):
        """Calculate NDVI"""
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
            ndvi[np.isnan(ndvi)] = 0
            return ndvi
    
    def _calculate_ndwi(self, green, swir):
        """Calculate NDWI"""
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = (green - swir) / (green + swir)
            ndwi[np.isnan(ndwi)] = 0
            return ndwi
    
    def quick_overview(self):
        """Create overview grid of all VRT files"""
        if not self.vrt_files:
            print("No VRT files found!")
            return
        
        print(f"\nCreating quick overview of {len(self.vrt_files)} files...")
        
        n_files = len(self.vrt_files)
        cols = 5
        rows = (n_files + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(25, rows * 4))
        fig.suptitle('Landsat 8/9 VRT Files - Natural Color Composites', 
                    fontsize=18, fontweight='bold')
        
        if n_files == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, vrt_file in enumerate(self.vrt_files):
            try:
                print(f"  Processing {idx+1}/{n_files}: {os.path.basename(vrt_file)}")
                
                # Read RGB bands (4, 3, 2 for natural color)
                if GDAL_AVAILABLE:
                    red, _ = self.read_band_gdal(vrt_file, 4)
                    green, _ = self.read_band_gdal(vrt_file, 3)
                    blue, _ = self.read_band_gdal(vrt_file, 2)
                else:
                    red, _ = self.read_band_rasterio(vrt_file, 4)
                    green, _ = self.read_band_rasterio(vrt_file, 3)
                    blue, _ = self.read_band_rasterio(vrt_file, 2)
                
                rgb = self._create_rgb_composite(red, green, blue)
                
                axes[idx].imshow(rgb)
                filename = os.path.basename(vrt_file).replace('Landsat8_', 'L8 ').replace('Landsat9_', 'L9 ').replace('.vrt', '')
                axes[idx].set_title(filename, fontsize=9)
                axes[idx].axis('off')
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                axes[idx].text(0.5, 0.5, f'Error\n{os.path.basename(vrt_file)}', 
                             ha='center', va='center')
                axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_files, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.vrt_directory, 'all_vrt_overview.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved overview to: {output_path}")
        
        plt.show()


def main():
    vrt_directory = r"F:\NIDM-SRM Drone Project Docs\L5_VRT"
    print("="*80)
    print("LANDSAT 8/9 VRT VISUALIZER")
    if GDAL_AVAILABLE:
        print("Using: GDAL (osgeo)")
    else:
        print("Using: Rasterio (GDAL wrapper)")
    print("="*80)
    
    visualizer = VRTVisualizerGDAL(vrt_directory)
    
    print("\nOptions:")
    print("1. Quick overview of all VRT files (grid view)")
    print("2. Detailed visualization (select specific files)")
    print("3. View information only")
    print("="*80)
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == '1':
        visualizer.quick_overview()
    elif choice == '2':
        print("\nAvailable VRT files:")
        for i, vrt in enumerate(visualizer.vrt_files, 1):
            print(f"{i:2d}. {os.path.basename(vrt)}")
        
        file_nums = input("\nEnter file numbers (comma-separated, or 'all'): ").strip()
        
        if file_nums.lower() == 'all':
            for vrt in visualizer.vrt_files:
                if GDAL_AVAILABLE:
                    visualizer.print_vrt_info_gdal(vrt)
                else:
                    visualizer.print_vrt_info_rasterio(vrt)
                visualizer.visualize_single_vrt(vrt, save_figures=True)
        else:
            for num_str in file_nums.split(','):
                try:
                    num = int(num_str.strip()) - 1
                    if 0 <= num < len(visualizer.vrt_files):
                        vrt = visualizer.vrt_files[num]
                        if GDAL_AVAILABLE:
                            visualizer.print_vrt_info_gdal(vrt)
                        else:
                            visualizer.print_vrt_info_rasterio(vrt)
                        visualizer.visualize_single_vrt(vrt, save_figure=True)
                except:
                    print(f"Invalid number: {num_str}")
    
    elif choice == '3':
        for vrt in visualizer.vrt_files:
            if GDAL_AVAILABLE:
                visualizer.print_vrt_info_gdal(vrt)
            else:
                visualizer.print_vrt_info_rasterio(vrt)
    else:
        print("\nDefaulting to quick overview...")
        visualizer.quick_overview()


if __name__ == "__main__":
    main()





