"""
Script to open and view all VRT files in the L5_VRT directory using GDAL (osgeo)
"""

import os
import glob
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Enable GDAL exceptions for better error handling
gdal.UseExceptions()

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
            dataset = gdal.Open(vrt_path, gdal.GA_ReadOnly)
            
            if dataset is None:
                print(f"ERROR: Could not open {vrt_path}")
                return None
            
            # Basic information
            print(f"\n📊 BASIC INFORMATION:")
            print(f"  Driver: {dataset.GetDriver().ShortName}/{dataset.GetDriver().LongName}")
            print(f"  Size: {dataset.RasterXSize} x {dataset.RasterYSize} pixels")
            print(f"  Number of Bands: {dataset.RasterCount}")
            
            # Projection and geotransform
            print(f"\n🌍 GEOSPATIAL INFORMATION:")
            projection = dataset.GetProjection()
            if projection:
                print(f"  Projection: {projection[:100]}..." if len(projection) > 100 else f"  Projection: {projection}")
            else:
                print(f"  Projection: Not defined")
            
            geotransform = dataset.GetGeoTransform()
            if geotransform:
                print(f"  GeoTransform:")
                print(f"    Origin: ({geotransform[0]:.2f}, {geotransform[3]:.2f})")
                print(f"    Pixel Size: ({geotransform[1]:.2f}, {geotransform[5]:.2f})")
                
                # Calculate extent
                ulx = geotransform[0]
                uly = geotransform[3]
                lrx = ulx + (geotransform[1] * dataset.RasterXSize)
                lry = uly + (geotransform[5] * dataset.RasterYSize)
                print(f"    Extent: ({ulx:.2f}, {lry:.2f}) to ({lrx:.2f}, {uly:.2f})")
            
            # Band information
            print(f"\n📡 BAND INFORMATION:")
            for i in range(1, dataset.RasterCount + 1):
                band = dataset.GetRasterBand(i)
                print(f"\n  Band {i}:")
                print(f"    Data Type: {gdal.GetDataTypeName(band.DataType)}")
                print(f"    Block Size: {band.GetBlockSize()}")
                print(f"    NoData Value: {band.GetNoDataValue()}")
                print(f"    Color Interpretation: {gdal.GetColorInterpretationName(band.GetColorInterpretation())}")
                
                # Get band statistics
                try:
                    stats = band.GetStatistics(True, True)
                    if stats:
                        print(f"    Statistics:")
                        print(f"      Min: {stats[0]:.2f}")
                        print(f"      Max: {stats[1]:.2f}")
                        print(f"      Mean: {stats[2]:.2f}")
                        print(f"      Std Dev: {stats[3]:.2f}")
                except:
                    print(f"    Statistics: Not available")
            
            # Metadata
            metadata = dataset.GetMetadata()
            if metadata:
                print(f"\n📝 METADATA:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
            
            return dataset
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
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
            dataset = gdal.Open(vrt_path, gdal.GA_ReadOnly)
            if dataset is None:
                print(f"ERROR: Could not open {vrt_path}")
                return
            
            if band_number > dataset.RasterCount:
                print(f"ERROR: Band {band_number} does not exist. File has {dataset.RasterCount} bands.")
                return
            
            band = dataset.GetRasterBand(band_number)
            data = band.ReadAsArray()
            
            # Create figure with subplots
            if show_histogram:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
            
            # Display the band data
            nodata = band.GetNoDataValue()
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
            dataset = gdal.Open(vrt_path, gdal.GA_ReadOnly)
            if dataset is None:
                print(f"ERROR: Could not open {vrt_path}")
                return
            
            if max(red_band, green_band, blue_band) > dataset.RasterCount:
                print(f"ERROR: One or more bands do not exist. File has {dataset.RasterCount} bands.")
                return
            
            # Read the bands
            red = dataset.GetRasterBand(red_band).ReadAsArray().astype(float)
            green = dataset.GetRasterBand(green_band).ReadAsArray().astype(float)
            blue = dataset.GetRasterBand(blue_band).ReadAsArray().astype(float)
            
            # Normalize to 0-1 range
            def normalize(band):
                band_min, band_max = np.percentile(band, [2, 98])
                normalized = (band - band_min) / (band_max - band_min)
                return np.clip(normalized, 0, 1)
            
            red_norm = normalize(red)
            green_norm = normalize(green)
            blue_norm = normalize(blue)
            
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
    
    def interactive_menu(self):
        """
        Interactive menu to explore VRT files
        """
        while True:
            print(f"\n{'='*80}")
            print("VRT VIEWER - INTERACTIVE MENU")
            print(f"{'='*80}")
            print("1. List all VRT files")
            print("2. View information for all VRT files")
            print("3. View information for a specific VRT file")
            print("4. View band data (grayscale)")
            print("5. View RGB composite")
            print("6. Exit")
            print(f"{'='*80}")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
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
                print("\nExiting VRT Viewer. Goodbye!")
                break
                
            else:
                print("\nInvalid choice! Please enter a number between 1 and 6.")


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


if __name__ == "__main__":
    main()





