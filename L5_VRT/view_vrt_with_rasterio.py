"""
Script to open and view all VRT files in the L5_VRT directory using Rasterio
(Rasterio is a Python wrapper for GDAL and easier to install)
"""

import os
import glob
import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class VRTViewer:
    def __init__(self, vrt_directory):
        """
        Initialize the VRT Viewer
        
        Args:
            vrt_directory (str): Path to the directory containing VRT files
        """
        self.vrt_directory = vrt_directory
        self.vrt_files = sorted(glob.glob(os.path.join(vrt_directory, "*.vrt")))
        
    def print_vrt_info(self, vrt_path):
        """
        Print detailed information about a VRT file
        
        Args:
            vrt_path (str): Path to the VRT file
        """
        print(f"\n{'='*80}")
        print(f"VRT File: {os.path.basename(vrt_path)}")
        print(f"{'='*80}")
        
        try:
            # Open the VRT file
            with rasterio.open(vrt_path) as src:
                
                # Basic information
                print(f"\n📊 BASIC INFORMATION:")
                print(f"  Driver: {src.driver}")
                print(f"  Size: {src.width} x {src.height} pixels")
                print(f"  Number of Bands: {src.count}")
                print(f"  Data Type: {src.dtypes[0]}")
                
                # Projection and bounds
                print(f"\n🌍 GEOSPATIAL INFORMATION:")
                print(f"  CRS: {src.crs}")
                print(f"  Bounds: {src.bounds}")
                print(f"  Transform:\n{src.transform}")
                
                # Calculate extent
                print(f"  Extent:")
                print(f"    Left: {src.bounds.left:.2f}")
                print(f"    Right: {src.bounds.right:.2f}")
                print(f"    Top: {src.bounds.top:.2f}")
                print(f"    Bottom: {src.bounds.bottom:.2f}")
                
                # Resolution
                print(f"  Resolution: {src.res[0]:.2f} x {src.res[1]:.2f}")
                
                # Band information
                print(f"\n📡 BAND INFORMATION:")
                for i in range(1, src.count + 1):
                    print(f"\n  Band {i}:")
                    print(f"    Data Type: {src.dtypes[i-1]}")
                    print(f"    NoData Value: {src.nodatavals[i-1]}")
                    
                    # Get band statistics
                    try:
                        band_data = src.read(i)
                        nodata = src.nodatavals[i-1]
                        
                        if nodata is not None:
                            valid_data = band_data[band_data != nodata]
                        else:
                            valid_data = band_data
                        
                        if len(valid_data) > 0:
                            print(f"    Statistics:")
                            print(f"      Min: {np.min(valid_data):.2f}")
                            print(f"      Max: {np.max(valid_data):.2f}")
                            print(f"      Mean: {np.mean(valid_data):.2f}")
                            print(f"      Std Dev: {np.std(valid_data):.2f}")
                        else:
                            print(f"    Statistics: No valid data")
                    except Exception as e:
                        print(f"    Statistics: Error reading data - {str(e)}")
                
                # Metadata
                metadata = src.meta
                if metadata:
                    print(f"\n📝 METADATA:")
                    for key, value in metadata.items():
                        print(f"  {key}: {value}")
                
                # Tags
                tags = src.tags()
                if tags:
                    print(f"\n🏷️  TAGS:")
                    for key, value in tags.items():
                        print(f"  {key}: {value}")
                
                return src.meta
                
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def view_band_data(self, vrt_path, band_number=1, show_histogram=False):
        """
        View a specific band from a VRT file
        
        Args:
            vrt_path (str): Path to the VRT file
            band_number (int): Band number to view (1-indexed)
            show_histogram (bool): Whether to show histogram
        """
        try:
            with rasterio.open(vrt_path) as src:
                if band_number > src.count:
                    print(f"ERROR: Band {band_number} does not exist. File has {src.count} bands.")
                    return
                
                # Read the band data
                data = src.read(band_number)
                nodata = src.nodatavals[band_number-1]
                
                # Create figure with subplots
                if show_histogram:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                else:
                    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
                
                # Display the band data
                if nodata is not None:
                    data_masked = np.ma.masked_equal(data, nodata)
                    im = ax1.imshow(data_masked, cmap='gray')
                else:
                    im = ax1.imshow(data, cmap='gray')
                
                ax1.set_title(f"{os.path.basename(vrt_path)}\nBand {band_number}")
                ax1.set_xlabel("Column (pixels)")
                ax1.set_ylabel("Row (pixels)")
                plt.colorbar(im, ax=ax1, label="Pixel Value")
                
                # Show histogram if requested
                if show_histogram:
                    if nodata is not None:
                        valid_data = data[data != nodata]
                    else:
                        valid_data = data.flatten()
                    
                    ax2.hist(valid_data, bins=50, edgecolor='black')
                    ax2.set_title("Value Distribution")
                    ax2.set_xlabel("Pixel Value")
                    ax2.set_ylabel("Frequency")
                    ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"ERROR viewing band data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def view_rgb_composite(self, vrt_path, red_band=3, green_band=2, blue_band=1):
        """
        Create and display an RGB composite from VRT file
        
        Args:
            vrt_path (str): Path to the VRT file
            red_band (int): Band to use for red channel
            green_band (int): Band to use for green channel
            blue_band (int): Band to use for blue channel
        """
        try:
            with rasterio.open(vrt_path) as src:
                if max(red_band, green_band, blue_band) > src.count:
                    print(f"ERROR: One or more bands do not exist. File has {src.count} bands.")
                    return
                
                # Read the bands
                red = src.read(red_band).astype(float)
                green = src.read(green_band).astype(float)
                blue = src.read(blue_band).astype(float)
                
                # Get nodata value
                nodata = src.nodatavals[0]
                
                # Normalize to 0-1 range
                def normalize(band, nodata):
                    if nodata is not None:
                        valid_data = band[band != nodata]
                    else:
                        valid_data = band
                    
                    if len(valid_data) > 0:
                        band_min, band_max = np.percentile(valid_data, [2, 98])
                        normalized = (band - band_min) / (band_max - band_min)
                        return np.clip(normalized, 0, 1)
                    else:
                        return band
                
                red_norm = normalize(red, nodata)
                green_norm = normalize(green, nodata)
                blue_norm = normalize(blue, nodata)
                
                # Stack into RGB
                rgb = np.dstack((red_norm, green_norm, blue_norm))
                
                # Display
                plt.figure(figsize=(12, 10))
                plt.imshow(rgb)
                plt.title(f"{os.path.basename(vrt_path)}\nRGB Composite (R={red_band}, G={green_band}, B={blue_band})")
                plt.xlabel("Column (pixels)")
                plt.ylabel("Row (pixels)")
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"ERROR creating RGB composite: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def list_all_vrt_files(self):
        """
        List all VRT files in the directory with basic info
        """
        print(f"\n{'='*80}")
        print(f"VRT FILES IN DIRECTORY: {self.vrt_directory}")
        print(f"{'='*80}\n")
        
        if not self.vrt_files:
            print("No VRT files found!")
            return
        
        print(f"Total VRT files found: {len(self.vrt_files)}\n")
        
        for i, vrt_file in enumerate(self.vrt_files, 1):
            filename = os.path.basename(vrt_file)
            file_size = os.path.getsize(vrt_file)
            print(f"{i:2d}. {filename:<35s} ({file_size:,} bytes)")
    
    def view_all_vrt_info(self):
        """
        Display information for all VRT files
        """
        self.list_all_vrt_files()
        
        for vrt_file in self.vrt_files:
            self.print_vrt_info(vrt_file)
    
    def compare_vrt_files(self):
        """
        Create a comparison table of all VRT files
        """
        print(f"\n{'='*80}")
        print("VRT FILES COMPARISON")
        print(f"{'='*80}\n")
        
        comparison_data = []
        
        for vrt_file in self.vrt_files:
            try:
                with rasterio.open(vrt_file) as src:
                    comparison_data.append({
                        'filename': os.path.basename(vrt_file),
                        'size': f"{src.width}x{src.height}",
                        'bands': src.count,
                        'dtype': src.dtypes[0],
                        'crs': str(src.crs)
                    })
            except Exception as e:
                comparison_data.append({
                    'filename': os.path.basename(vrt_file),
                    'size': 'ERROR',
                    'bands': 'ERROR',
                    'dtype': 'ERROR',
                    'crs': str(e)
                })
        
        # Print comparison table
        print(f"{'Filename':<35s} {'Size':<15s} {'Bands':<7s} {'Data Type':<12s}")
        print("-" * 80)
        for data in comparison_data:
            print(f"{data['filename']:<35s} {data['size']:<15s} {data['bands']:<7} {data['dtype']:<12s}")
    
    def interactive_menu(self):
        """
        Interactive menu to explore VRT files
        """
        while True:
            print(f"\n{'='*80}")
            print("VRT VIEWER - INTERACTIVE MENU (Using Rasterio)")
            print(f"{'='*80}")
            print("1. List all VRT files")
            print("2. View information for all VRT files")
            print("3. View information for a specific VRT file")
            print("4. View band data (grayscale)")
            print("5. View RGB composite")
            print("6. Compare all VRT files")
            print("7. Exit")
            print(f"{'='*80}")
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                self.list_all_vrt_files()
                
            elif choice == '2':
                self.view_all_vrt_info()
                
            elif choice == '3':
                self.list_all_vrt_files()
                try:
                    file_idx = int(input("\nEnter file number: ")) - 1
                    if 0 <= file_idx < len(self.vrt_files):
                        self.print_vrt_info(self.vrt_files[file_idx])
                    else:
                        print("Invalid file number!")
                except ValueError:
                    print("Invalid input!")
                    
            elif choice == '4':
                self.list_all_vrt_files()
                try:
                    file_idx = int(input("\nEnter file number: ")) - 1
                    if 0 <= file_idx < len(self.vrt_files):
                        band_num = int(input("Enter band number (1-7): "))
                        show_hist = input("Show histogram? (y/n): ").strip().lower() == 'y'
                        self.view_band_data(self.vrt_files[file_idx], band_num, show_hist)
                    else:
                        print("Invalid file number!")
                except ValueError:
                    print("Invalid input!")
                    
            elif choice == '5':
                self.list_all_vrt_files()
                try:
                    file_idx = int(input("\nEnter file number: ")) - 1
                    if 0 <= file_idx < len(self.vrt_files):
                        print("\nDefault: Red=3, Green=2, Blue=1")
                        use_default = input("Use default? (y/n): ").strip().lower() == 'y'
                        if use_default:
                            self.view_rgb_composite(self.vrt_files[file_idx])
                        else:
                            r = int(input("Red band: "))
                            g = int(input("Green band: "))
                            b = int(input("Blue band: "))
                            self.view_rgb_composite(self.vrt_files[file_idx], r, g, b)
                    else:
                        print("Invalid file number!")
                except ValueError:
                    print("Invalid input!")
                    
            elif choice == '6':
                self.compare_vrt_files()
                
            elif choice == '7':
                print("\nExiting VRT Viewer. Goodbye!")
                break
                
            else:
                print("\nInvalid choice! Please enter a number between 1 and 7.")


def main():
    """
    Main function to run the VRT viewer
    """
    # Define the VRT directory
    vrt_directory = r"f:\NIDM-SRM Drone Project Docs\L5_VRT"
    
    # Create viewer instance
    viewer = VRTViewer(vrt_directory)
    
    # Run interactive menu
    viewer.interactive_menu()
    
    # Alternative: View all VRT information directly (uncomment to use)
    # viewer.view_all_vrt_info()
    
    # Alternative: Just compare all files (uncomment to use)
    # viewer.compare_vrt_files()


if __name__ == "__main__":
    main()





