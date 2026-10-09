"""
VRT Diagnostic Tool - Shows what files VRT files are referencing
"""

import os
import glob
import xml.etree.ElementTree as ET

def analyze_vrt_file(vrt_path):
    """Analyze a VRT file and extract information"""
    try:
        tree = ET.parse(vrt_path)
        root = tree.getroot()
        
        info = {
            'filename': os.path.basename(vrt_path),
            'raster_size': f"{root.get('rasterXSize')} x {root.get('rasterYSize')}",
            'bands': len(root.findall('.//VRTRasterBand')),
            'referenced_files': []
        }
        
        # Find all source filenames
        for source_filename in root.findall('.//SourceFilename'):
            filename = source_filename.text
            is_relative = source_filename.get('relativeToVRT') == '1'
            
            if is_relative:
                full_path = os.path.join(os.path.dirname(vrt_path), filename)
            else:
                full_path = filename
            
            exists = os.path.exists(full_path)
            
            info['referenced_files'].append({
                'name': filename,
                'full_path': full_path,
                'relative': is_relative,
                'exists': exists
            })
        
        # Remove duplicates
        unique_files = {}
        for ref in info['referenced_files']:
            unique_files[ref['name']] = ref
        info['referenced_files'] = list(unique_files.values())
        
        # Get projection info
        srs = root.find('.//SRS')
        if srs is not None:
            info['projection'] = srs.text[:100] if srs.text else 'Unknown'
        else:
            info['projection'] = 'Unknown'
        
        # Get geotransform
        geotransform = root.find('.//GeoTransform')
        if geotransform is not None:
            info['geotransform'] = geotransform.text.strip() if geotransform.text else 'Unknown'
        else:
            info['geotransform'] = 'Unknown'
        
        return info
        
    except Exception as e:
        return {
            'filename': os.path.basename(vrt_path),
            'error': str(e)
        }


def main():
    vrt_directory = r"f:\NIDM-SRM Drone Project Docs\L5_VRT"
    
    print("="*80)
    print("VRT FILE DIAGNOSTIC TOOL - LANDSAT 5")
    print("="*80)
    print(f"\nAnalyzing VRT files in: {vrt_directory}\n")
    
    vrt_files = sorted(glob.glob(os.path.join(vrt_directory, "*.vrt")))
    
    if not vrt_files:
        print("No VRT files found!")
        return
    
    print(f"Found {len(vrt_files)} VRT files\n")
    print("="*80)
    
    missing_files = set()
    all_info = []
    
    for i, vrt_file in enumerate(vrt_files, 1):
        info = analyze_vrt_file(vrt_file)
        all_info.append(info)
        
        if 'error' in info:
            print(f"\n[{i}] {info['filename']}")
            print(f"  ERROR: {info['error']}")
            continue
        
        print(f"\n[{i}] {info['filename']}")
        print(f"  Size: {info['raster_size']}")
        print(f"  Bands: {info['bands']}")
        print(f"  Referenced files:")
        
        for ref in info['referenced_files']:
            status = "[OK]" if ref['exists'] else "[MISSING]"
            print(f"    - {ref['name']:40s} {status}")
            if not ref['exists']:
                missing_files.add(ref['name'])
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total VRT files analyzed: {len(all_info)}")
    print(f"Unique missing TIFF files: {len(missing_files)}")
    
    if missing_files:
        print(f"\n{'='*80}")
        print("MISSING FILES LIST:")
        print(f"{'='*80}")
        for filename in sorted(missing_files):
            print(f"  - {filename}")
        
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS:")
        print(f"{'='*80}")
        print("1. VRT files are 'Virtual Raster' files - they reference actual TIFF files")
        print("2. The TIFF files need to be in the same directory as the VRT files")
        print("3. You need to obtain the actual Landsat 5 imagery TIFF files")
        print("4. Common sources:")
        print("   - USGS EarthExplorer (https://earthexplorer.usgs.gov/)")
        print("   - Google Earth Engine")
        print("   - AWS Open Data (Landsat on AWS)")
        print(f"\n5. Expected location: {vrt_directory}")
        print(f"\n6. Once you have the TIFF files, place them in the VRT directory")
        print("   and the visualization scripts will work properly.")
    else:
        print("\nAll referenced files exist! You can proceed with visualization.")
    
    # Save report to file
    report_path = os.path.join(vrt_directory, "vrt_diagnostic_report.txt")
    with open(report_path, 'w') as f:
        f.write("VRT DIAGNOSTIC REPORT - LANDSAT 5\n")
        f.write("="*80 + "\n\n")
        f.write(f"Directory: {vrt_directory}\n")
        f.write(f"Total VRT files: {len(vrt_files)}\n")
        f.write(f"Missing TIFF files: {len(missing_files)}\n\n")
        
        f.write("MISSING FILES:\n")
        f.write("-"*80 + "\n")
        for filename in sorted(missing_files):
            f.write(f"{filename}\n")
    
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()





