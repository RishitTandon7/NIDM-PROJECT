"""
Update VRT source filenames to reference .vrt files instead of .tif files
This is useful when you have a hierarchy of VRT files
"""
import os
import re
from pathlib import Path


def update_vrt_to_vrt_references(vrt_file, new_base_path=None):
    """
    Update a VRT file to reference .vrt files instead of .tif files.
    
    Args:
        vrt_file: Path to the VRT file to update
        new_base_path: Optional new base directory path
    """
    
    with open(vrt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Find all SourceFilename tags
    pattern = r'(\u003cSourceFilename[^>]*\u003e)([^\u003c]+)(\u003c/SourceFilename\u003e)'
    
    def replace_extension(match):
        prefix = match.group(1)
        filepath = match.group(2)
        suffix = match.group(3)
        
        # Change .tif to .vrt
        if filepath.endswith('.tif') or filepath.endswith('.TIF'):
            filepath = filepath.rsplit('.', 1)[0] + '.vrt'
        
        # Optionally update the base path
        if new_base_path:
            filename = os.path.basename(filepath)
            filepath = os.path.join(new_base_path, filename)
        
        return prefix + filepath + suffix
    
    content = re.sub(pattern, replace_extension, content)
    
    # Only write if changes were made
    if content != original_content:
        with open(vrt_file, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False


def batch_update_vrt_directory(vrt_dir, new_base_path=None):
    """
    Update all VRT files in a directory.
    
    Args:
        vrt_dir: Directory containing VRT files
        new_base_path: Optional new base path for source files
    """
    
    print(f"\n{'='*70}")
    print(f" Updating VRT files in: {vrt_dir}")
    print(f"{'='*70}\n")
    
    if new_base_path:
        print(f"New base path: {new_base_path}\n")
    
    vrt_files = list(Path(vrt_dir).glob('*.vrt'))
    
    if not vrt_files:
        print("No VRT files found!")
        return
    
    print(f"Found {len(vrt_files)} VRT files\n")
    
    updated_count = 0
    
    for vrt_file in vrt_files:
        try:
            was_updated = update_vrt_to_vrt_references(str(vrt_file), new_base_path)
            
            if was_updated:
                print(f"[OK] Updated: {vrt_file.name}")
                updated_count += 1
            else:
                print(f"[SKIP] No changes needed: {vrt_file.name}")
        
        except Exception as e:
            print(f"[ERROR] Failed to update {vrt_file.name}: {str(e)}")
    
    print(f"\n{'='*70}")
    print(f" Updated {updated_count} / {len(vrt_files)} files")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    
    # Directories to process
    vrt_dirs = [
        os.path.join(base_dir, 'L5_VRT'),
        os.path.join(base_dir, 'L8_9_VRT-20251212T092651Z-1-001', 'L8_9_VRT'),
        os.path.join(base_dir, 'S2_VRT-20251212T092639Z-1-001', 'S2_VRT'),
        os.path.join(base_dir, 'Liss4Clip_VRT-20251212T092648Z-1-001', 'Liss4Clip_VRT')
    ]
    
    # Option 1: Just change .tif to .vrt (keep same path)
    print("\nOption 1: Change .tif extensions to .vrt (keep existing paths)")
    print("="*70)
    
    for vrt_dir in vrt_dirs:
        if os.path.exists(vrt_dir):
            batch_update_vrt_directory(vrt_dir, new_base_path=None)
    
    # Option 2: Change extensions AND update base path
    # Uncomment below if you also want to change the directory path
    
    # new_base = r"f:\NIDM-SRM Drone Project Docs\VRT_Data"  # Your new location
    # for vrt_dir in vrt_dirs:
    #     if os.path.exists(vrt_dir):
    #         batch_update_vrt_directory(vrt_dir, new_base_path=new_base)
    
    print("\n[COMPLETE] All VRT files have been processed!")
    print("\nNOTE: VRT files now reference .vrt files instead of .tif files")
    print("Make sure the referenced .vrt files exist at the specified locations!")





