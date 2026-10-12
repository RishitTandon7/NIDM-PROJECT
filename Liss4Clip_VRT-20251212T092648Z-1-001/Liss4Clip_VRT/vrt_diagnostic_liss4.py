"""
VRT Diagnostic Tool for LISS-4 files
Analyzes nested VRT references
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
        
        # Find all SourceDataset elements (for warped VRT)
        for source_dataset in root.findall('.//SourceDataset'):
            filename = source_dataset.text
            is_relative = source_dataset.get('relativeToVRT') == '1'
            
            if is_relative:
                full_path = os.path.join(os.path.dirname(vrt_path), filename)
            else:
                full_path = filename
            
            exists = os.path.exists(full_path)
            
            info['referenced_files'].append({
                'name': filename,
                'full_path': full_path,
                'relative': is_relative,
                'exists': exists,
                'type': 'VRT (nested)'
            })
        
        # Find all SourceFilename elements (for regular VRT)
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
                'exists': exists,
                'type': 'TIFF'
            })
        
        # Remove duplicates
        unique_files = {}
        for ref in info['referenced_files']:
            unique_files[ref['name']] = ref
        info['referenced_files'] = list(unique_files.values())
        
        return info
        
    except Exception as e:
        return {
            'filename': os.path.basename(vrt_path),
            'error': str(e)
        }


def main():
    vrt_directory = r"f:\NIDM-SRM Drone Project Docs\Liss4Clip_VRT-20251212T092648Z-1-001\Liss4Clip_VRT"
    
    print("="*80)
    print("VRT FILE DIAGNOSTIC TOOL - LISS-4")
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
    
    # Analyze first 3 files as sample
    for i, vrt_file in enumerate(vrt_files[:3], 1):
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
            print(f"    - {os.path.basename(ref['name']):50s} {ref['type']:15s} {status}")
            if not ref['exists']:
                missing_files.add(ref['name'])
    
    print(f"\n... (showing 3 of {len(vrt_files)} files)")
    
    # Analyze all to get complete picture
    print(f"\nAnalyzing all {len(vrt_files)} files for missing references...")
    for vrt_file in vrt_files:
        info = analyze_vrt_file(vrt_file)
        if 'error' not in info:
            for ref in info['referenced_files']:
                if not ref['exists']:
                    missing_files.add(ref['name'])
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total VRT files analyzed: {len(vrt_files)}")
    print(f"Unique missing files: {len(missing_files)}")
    
    if missing_files:
        print(f"\n{'='*80}")
        print("MISSING FILES (Sample):")
        print(f"{'='*80}")
        for filename in sorted(list(missing_files))[:10]:
            print(f"  - {filename}")
        
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        
        print(f"\n{'='*80}")
        print("ANALYSIS:")
        print(f"{'='*80}")
        print("These VRT files are 'clipped' versions that reference other VRT files")
        print("from your professor's laptop (D:\\GLOF Liss 4\\Liss4_VRT\\)")
        print("\nThese intermediate VRT files are also missing from your system.")
        print("\nYou need EITHER:")
        print("  1. The original LISS-4 TIFF files")
        print("  2. The intermediate VRT files from D:\\GLOF Liss 4\\Liss4_VRT\\")
        print("\nPlease check if you have the LISS-4 data stored elsewhere.")
    
    # Save report
    report_path = os.path.join(vrt_directory, "vrt_diagnostic_report.txt")
    with open(report_path, 'w') as f:
        f.write("VRT DIAGNOSTIC REPORT - LISS-4\n")
        f.write("="*80 + "\n\n")
        f.write(f"Directory: {vrt_directory}\n")
        f.write(f"Total VRT files: {len(vrt_files)}\n")
        f.write(f"Missing files: {len(missing_files)}\n\n")
        
        f.write("MISSING FILES:\n")
        f.write("-"*80 + "\n")
        for filename in sorted(missing_files):
            f.write(f"{filename}\n")
    
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()





