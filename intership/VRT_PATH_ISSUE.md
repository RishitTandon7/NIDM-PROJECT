# VRT Source File Path Issue - Summary

## Problem Identified

Your VRT files reference source TIFF files that don't exist at the specified locations:

### Current Paths in VRT Files:
- **Landsat 5 VRTs (17 files)**: `F:\CTECH\Research\proposal\NIDM\GLOF data\Landsat 5\*.tif`
- **Landsat 8/9 VRTs (31 files)**: `F:\CTECH\Research\proposal\NIDM\GLOF data\Landsat 8\*.tif`
- **Sentinel-2 VRTs (66 files)**: `F:\CTECH\Research\proposal\NIDM\GLOF data\Sentinel 2\*.tif`
- **LISS-IV VRTs (54 files)**: Using a different VRT structure

## What Are VRT Files?

VRT (Virtual Raster) files are **XML files** that act as pointers to actual raster data. They don't contain image data themselves, just metadata and references to where the real data is located.

**Example VRT structure:**
```xml
<VRTDataset rasterXSize="605" rasterYSize="505">
  <VRTRasterBand dataType="Int16" band="1">
    <SourceFilename relativeToVRT="0">
      F:\CTECH\Research\proposal\NIDM\GLOF data\Landsat 5\Landsat5_2009-03-14.tif
    </SourceFilename>
  </VRTRasterBand>
</VRTDataset>
```

## Solutions

### Solution 1: Locate Original TIFF Files ✅

If you have the original Landsat, Sentinel-2, and LISS-IV TIFF files somewhere on your system:

1. **Run the VRT path updater script:**
   ```powershell
   cd "f:\NIDM-SRM Drone Project Docs\intership"
   python fix_vrt_paths.py
   ```

2. **Choose option 2** to update to a new absolute path

3. **Enter the directory** where your source TIFF files are located

4. The script will update all VRT files to point to the correct location

### Solution 2: Use Existing Data Folder (CURRENT APPROACH) ✅

This is what we've already implemented! The training scripts use the TIFF files from:
```
f:\NIDM-SRM Drone Project Docs\Data\
  ├── L8_image.tif (Landsat 8)
  ├── S2_image.tif (Sentinel-2)
  └── LISS_image.tif (LISS-IV)
```

**Scripts already working with this solution:**
- [train_satellite_data.py](file:///f:/NIDM-SRM%20Drone%20Project%20Docs/intership/train_satellite_data.py) ✓ Successfully trained models
- [read_satellite_images.py](file:///f:/NIDM-SRM%20Drone%20Project%20Docs/intership/read_satellite_images.py) ✓ Successfully read images

### Solution 3: Convert to Relative Paths

If you place the source TIFF files in the same directories as the VRT files:

1. Copy/move source TIFFs to VRT directories
2. Run `python fix_vrt_paths.py`
3. Choose option 3 to convert to relative paths
4. VRTs will now reference files in the same directory

## Created Tool

**[fix_vrt_paths.py](file:///f:/NIDM-SRM%20Drone%20Project%20Docs/intership/fix_vrt_paths.py)** - Interactive script that can:
- **Analyze** current VRT paths
- **Update** VRTs to new absolute paths  
- **Convert** VRTs to use relative paths

## Recommendation

Since the original source files from `F:\CTECH\Research\proposal\NIDM\` are not available on your system, I recommend:

1. **Continue using the Data folder approach** (already working!)
2. **If you need to use the VRT files:**
   - Locate where the original Landsat 5, 8, 9, Sentinel-2, and LISS-IV TIFF files are stored
   - Run `fix_vrt_paths.py` to update all VRT references
   - Or copy the source TIFFs into the VRT directories and convert to relative paths

## Benefits of VRT Files (When Working)

- **Time-series analysis**: All dates in one directory
- **Virtual mosaics**: Combine multiple tiles without creating new files
- **Save disk space**: No need to duplicate or convert files
- **Quick subsetting**: Work with portions of large datasets

For now, the training pipeline works perfectly with the direct TIFF files in the Data folder!
