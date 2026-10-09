"""
Fix VRT files to point to local .tif files
This prepares the VRTs so they work immediately when you copy the .tif files into the folder
"""
import os
import re
from pathlib import Path


def set_vrt_to_local_tif(vrt_file):
    """
    Update VRT to reference .tif files in the same directory.
    Removes absolute paths and sets extension to .tif
    """
    
    with open(vrt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to find SourceFilename tags
    pattern = r'(<SourceFilename[^>]*>)([^<]+)(</SourceFilename>)'
    
    def replace_path(match):
        prefix = match.group(1)
        # We don't care what the old path was, we want to construct the new one
        filepath = match.group(2)
        suffix = match.group(3)
        
        # Get just the filename
        filename = os.path.basename(filepath)
        
        # Force .tif extension (remove .vrt or other extensions)
        base_name = filename.rsplit('.', 1)[0]
        new_filename = base_name + '.tif'
        
        # Use relative path (just the filename)
        # This tells GDAL to look in the same directory as the VRT
        return f'{prefix}{new_filename}{suffix}'
    
    content = re.sub(pattern, replace_path, content)
    
    # Also fix relativeToVRT attribute to be "1" (true)
    content = content.replace('relativeToVRT="0"', 'relativeToVRT="1"')
    
    if content != original_content:
        with open(vrt_file, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False


def batch_fix_directory(vrt_dir):
    """Update all VRT files in a directory."""
    
    vrt_files = list(Path(vrt_dir).glob('*.vrt'))
    
    if not vrt_files:
        print(f"[SKIP] No VRTs in {vrt_dir}")
        return 0
    
    print(f"\nProcessing {vrt_dir}...")
    updated_count = 0
    
    for vrt_file in vrt_files:
        try:
            if set_vrt_to_local_tif(str(vrt_file)):
                updated_count += 1
        except Exception as e:
            print(f"  [ERROR] {vrt_file.name}: {e}")
            
    print(f"  Updated {updated_count} files to point to local .tif")
    return updated_count


if __name__ == "__main__":
    print("="*70)
    print(" FIXING VRT FILES TO POINT TO LOCAL TIFFs")
    print("="*70)
    
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    
    dirs_to_fix = [
        os.path.join(base_dir, 'L5_VRT'),
        os.path.join(base_dir, 'L8_9_VRT-20251212T092651Z-1-001', 'L8_9_VRT'),
        os.path.join(base_dir, 'S2_VRT-20251212T092639Z-1-001', 'S2_VRT'),
        os.path.join(base_dir, 'Liss4Clip_VRT-20251212T092648Z-1-001', 'Liss4Clip_VRT')
    ]
    
    total = 0
    for d in dirs_to_fix:
        if os.path.exists(d):
            total += batch_fix_directory(d)
            
    print("\n" + "="*70)
    print(f" DONE: Updated {total} VRT files")
    print("="*70)
    print(" ACTION REQUIRED: Please copy your .tif files into these folders!")





