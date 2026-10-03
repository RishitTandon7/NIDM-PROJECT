"""
Script to update Sentinel-2 VRT file paths to point to the correct TIFF file location
"""

import os
import glob
import xml.etree.ElementTree as ET

def update_vrt_file(vrt_path, new_tif_directory):
    """Update a VRT file to point to new TIFF location"""
    try:
        # Parse the VRT file
        tree = ET.parse(vrt_path)
        root = tree.getroot()
        
        # Find all SourceFilename elements
        updated = False
        for source_filename in root.findall('.//SourceFilename'):
            old_filename = source_filename.text
            tif_basename = os.path.basename(old_filename)
            
            # Create new path (parent directory of VRT folder)
            new_tif_path = os.path.join(new_tif_directory, tif_basename)
            
            # Check if the new file exists
            if os.path.exists(new_tif_path):
                # Update to absolute path
                source_filename.text = new_tif_path
                source_filename.set('relativeToVRT', '0')
                updated = True
                print(f"  [OK] Updated: {tif_basename}")
            else:
                print(f"  [WARNING] {tif_basename} not found in {new_tif_directory}")
        
        if updated:
            # Write the updated VRT file
            tree.write(vrt_path, encoding='utf-8', xml_declaration=True)
            return True
        else:
            return False
            
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return False


def main():
    # Directories
    vrt_directory = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"
    tif_directory = r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2"
    
    print("="*80)
    print("VRT PATH UPDATER - SENTINEL-2")
    print("="*80)
    print(f"\nVRT Directory: {vrt_directory}")
    print(f"TIFF Directory: {tif_directory}\n")
    
    # Check if TIFF directory exists
    if not os.path.exists(tif_directory):
        print(f"ERROR: TIFF directory not found: {tif_directory}")
        return
    
    # Check if VRT directory exists
    if not os.path.exists(vrt_directory):
        print(f"ERROR: VRT directory not found: {vrt_directory}")
        return
    
    # Get all VRT files
    vrt_files = sorted(glob.glob(os.path.join(vrt_directory, "*.vrt")))
    
    if not vrt_files:
        print("No VRT files found!")
        return
    
    print(f"Found {len(vrt_files)} VRT files")
    print("="*80 + "\n")
    
    # Update each VRT file
    success_count = 0
    for i, vrt_file in enumerate(vrt_files, 1):
        print(f"[{i}/{len(vrt_files)}] {os.path.basename(vrt_file)}")
        
        if update_vrt_file(vrt_file, tif_directory):
            success_count += 1
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total VRT files: {len(vrt_files)}")
    print(f"Successfully updated: {success_count}")
    print(f"Failed: {len(vrt_files) - success_count}")
    
    if success_count == len(vrt_files):
        print("\n[SUCCESS] All VRT files successfully updated!")
        print("\nYou can now run the visualization scripts to view the Sentinel-2 imagery.")
    else:
        print("\n[WARNING] Some VRT files could not be updated. Check the output above for details.")


if __name__ == "__main__":
    main()





