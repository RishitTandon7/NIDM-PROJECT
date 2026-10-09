"""
Script to visualize all VRT files in the L5_VRT directory using Rasterio
Generates visual plots for each VRT file
"""

import os
import glob
import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set UTF-8 encoding for Windows console
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

class VRTVisualizer:
    def __init__(self, vrt_directory):
        self.vrt_directory = vrt_directory
        self.vrt_files = sorted(glob.glob(os.path.join(vrt_directory, "*.vrt")))
        
    def print_vrt_info(self, vrt_path):
        """Print detailed information about a VRT file"""
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
                for i in range(1, min(src.count + 1, 8)):  # Show up to 7 bands
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
    
    def visualize_single_vrt(self, vrt_path, save_figure=True):
        """Create comprehensive visualization for a single VRT file"""
        try:
            with rasterio.open(vrt_path) as src:
                filename = os.path.basename(vrt_path).replace('.vrt', '')
                
                # Create figure with multiple subplots
                fig = plt.figure(figsize=(20, 12))
                gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
                
                # Main title
                fig.suptitle(f'Landsat 5 Data Visualization: {filename}', 
                           fontsize=16, fontweight='bold')
                
                # Read all bands
                bands_data = []
                nodata = src.nodatavals[0]
                
                for i in range(1, min(src.count + 1, 8)):
                    band = src.read(i).astype(float)
                    if nodata is not None:
                        band[band == nodata] = np.nan
                    bands_data.append(band)
                
                # Plot individual bands (first 7)
                band_names = ['Band 1 (Blue)', 'Band 2 (Green)', 'Band 3 (Red)', 
                            'Band 4 (NIR)', 'Band 5 (SWIR1)', 'Band 6 (TIR)', 'Band 7 (SWIR2)']
                
                for idx in range(min(7, len(bands_data))):
                    ax = fig.add_subplot(gs[idx // 4, idx % 4])
                    
                    im = ax.imshow(bands_data[idx], cmap='gray')
                    ax.set_title(band_names[idx], fontsize=10)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # RGB Composite (Bands 3, 2, 1 - Natural Color)
                if len(bands_data) >= 3:
                    ax_rgb = fig.add_subplot(gs[1, 3])
                    rgb = self._create_rgb_composite(bands_data[2], bands_data[1], bands_data[0])
                    ax_rgb.imshow(rgb)
                    ax_rgb.set_title('Natural Color RGB (3,2,1)', fontsize=10)
                    ax_rgb.axis('off')
                
                # False Color Composite (Bands 4, 3, 2 - Vegetation)
                if len(bands_data) >= 4:
                    ax_false = fig.add_subplot(gs[2, 0])
                    false_color = self._create_rgb_composite(bands_data[3], bands_data[2], bands_data[1])
                    ax_false.imshow(false_color)
                    ax_false.set_title('False Color (4,3,2) - Vegetation', fontsize=10)
                    ax_false.axis('off')
                
                # SWIR Composite (Bands 7, 5, 3)
                if len(bands_data) >= 7:
                    ax_swir = fig.add_subplot(gs[2, 1])
                    swir = self._create_rgb_composite(bands_data[6], bands_data[4], bands_data[2])
                    ax_swir.imshow(swir)
                    ax_swir.set_title('SWIR Composite (7,5,3)', fontsize=10)
                    ax_swir.axis('off')
                
                # NDVI Calculation (if NIR and Red bands available)
                if len(bands_data) >= 4:
                    ax_ndvi = fig.add_subplot(gs[2, 2])
                    ndvi = self._calculate_ndvi(bands_data[3], bands_data[2])
                    im_ndvi = ax_ndvi.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
                    ax_ndvi.set_title('NDVI (Vegetation Index)', fontsize=10)
                    ax_ndvi.axis('off')
                    plt.colorbar(im_ndvi, ax=ax_ndvi, fraction=0.046, pad=0.04)
                
                # Histogram of all bands
                ax_hist = fig.add_subplot(gs[2, 3])
                for idx, band in enumerate(bands_data[:7]):
                    valid_data = band[~np.isnan(band)].flatten()
                    if len(valid_data) > 0:
                        ax_hist.hist(valid_data, bins=50, alpha=0.5, 
                                   label=f'Band {idx+1}', density=True)
                ax_hist.set_title('Band Histograms', fontsize=10)
                ax_hist.set_xlabel('Pixel Value')
                ax_hist.set_ylabel('Density')
                ax_hist.legend(fontsize=8)
                ax_hist.grid(True, alpha=0.3)
                
                if save_figure:
                    output_path = os.path.join(self.vrt_directory, f'{filename}_visualization.png')
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    print(f"Saved visualization to: {output_path}")
                
                plt.show()
                
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
        """Calculate NDVI (Normalized Difference Vegetation Index)"""
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
            ndvi[np.isnan(ndvi)] = 0
            return ndvi
    
    def visualize_all_vrt_files(self, save_figures=True):
        """Visualize all VRT files"""
        print(f"\n{'='*80}")
        print(f"Visualizing {len(self.vrt_files)} VRT files from: {self.vrt_directory}")
        print(f"{'='*80}\n")
        
        for i, vrt_file in enumerate(self.vrt_files, 1):
            print(f"\n[{i}/{len(self.vrt_files)}] Processing: {os.path.basename(vrt_file)}")
            self.print_vrt_info(vrt_file)
            
            # Ask user if they want to see each visualization
            response = input(f"\nVisualize this file? (y/n/all): ").strip().lower()
            
            if response == 'all':
                # Visualize all remaining files
                for remaining_file in self.vrt_files[i-1:]:
                    print(f"\nVisualizing: {os.path.basename(remaining_file)}")
                    self.visualize_single_vrt(remaining_file, save_figures)
                break
            elif response == 'y':
                self.visualize_single_vrt(vrt_file, save_figures)
            elif response == 'n':
                print("Skipped.")
                continue
            else:
                print("Invalid input. Skipped.")
    
    def quick_overview(self):
        """Create a quick overview of all VRT files"""
        print(f"\n{'='*80}")
        print(f"QUICK OVERVIEW - All VRT Files")
        print(f"{'='*80}\n")
        
        # Create a grid of RGB composites
        n_files = len(self.vrt_files)
        cols = 4
        rows = (n_files + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
        fig.suptitle('Landsat 5 VRT Files - Natural Color Composites', fontsize=16, fontweight='bold')
        
        if n_files == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, vrt_file in enumerate(self.vrt_files):
            try:
                with rasterio.open(vrt_file) as src:
                    if src.count >= 3:
                        red = src.read(3).astype(float)
                        green = src.read(2).astype(float)
                        blue = src.read(1).astype(float)
                        
                        nodata = src.nodatavals[0]
                        if nodata is not None:
                            red[red == nodata] = np.nan
                            green[green == nodata] = np.nan
                            blue[blue == nodata] = np.nan
                        
                        rgb = self._create_rgb_composite(red, green, blue)
                        
                        axes[idx].imshow(rgb)
                        filename = os.path.basename(vrt_file).replace('Landsat5_', '').replace('.vrt', '')
                        axes[idx].set_title(filename, fontsize=10)
                        axes[idx].axis('off')
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Error\n{os.path.basename(vrt_file)}', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_files, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.vrt_directory, 'all_vrt_overview.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved overview to: {output_path}")
        
        plt.show()


def main():
    vrt_directory = r"L5_VRT"
    
    visualizer = VRTVisualizer(vrt_directory)
    
    print("\n" + "="*80)
    print("LANDSAT 5 VRT VISUALIZER")
    print("="*80)
    print("\nOptions:")
    print("1. Quick overview of all VRT files (grid view)")
    print("2. Detailed visualization (one by one)")
    print("3. Visualize a specific file")
    print("="*80)
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == '1':
        visualizer.quick_overview()
    elif choice == '2':
        visualizer.visualize_all_vrt_files(save_figures=True)
    elif choice == '3':
        print("\nAvailable VRT files:")
        for i, vrt in enumerate(visualizer.vrt_files, 1):
            print(f"{i}. {os.path.basename(vrt)}")
        
        file_num = int(input("\nEnter file number: ")) - 1
        if 0 <= file_num < len(visualizer.vrt_files):
            visualizer.print_vrt_info(visualizer.vrt_files[file_num])
            visualizer.visualize_single_vrt(visualizer.vrt_files[file_num], save_figures=True)
        else:
            print("Invalid file number!")
    else:
        print("\nDefaulting to quick overview...")
        visualizer.quick_overview()


if __name__ == "__main__":
    main()





