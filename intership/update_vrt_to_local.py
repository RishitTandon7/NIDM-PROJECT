"""
Update VRT files to reference .vrt files in their own directory
"""
import os
import re
from pathlib import Path


def update_vrt_to_local_references(vrt_file):
    """
    Update VRT to reference .vrt files in the same directory.
    Changes both the path and extension.
    """
    
    vrt_dir = os.path.dirname(vrt_file)
    
    with open(vrt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to find SourceFilename tags
    pattern = r'(<SourceFilename[^>]*>)([^<]+)(</SourceFilename>)'
    
    def replace_path(match):
        prefix = match.group(1)
        filepath = match.group(2)
        suffix = match.group(3)
        
        # Get just the filename from the current path
        filename = os.path.basename(filepath)
        
        # Change extension from .tif to .vrt if needed
        if filename.endswith('.tif') or filename.endswith('.TIF'):
            filename = filename.rsplit('.', 1)[0] + '.vrt'
        
        # Create new path in the same directory as the VRT file
        new_path = os.path.join(vrt_dir, filename)
        
        return prefix + new_path + suffix
    
    content = re.sub(pattern, replace_path, content)
    
    # Only write if changes were made
    if content != original_content:
        with open(vrt_file, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False


def batch_update_directory(vrt_dir):
    """Update all VRT files in a directory."""
    
    print(f"\n{'='*70}")
    print(f" Updating: {vrt_dir}")
    print(f"{'='*70}\n")
    
    vrt_files = list(Path(vrt_dir).glob('*.vrt'))
    
    if not vrt_files:
        print("No VRT files found!")
        return
    
    print(f"Found {len(vrt_files)} VRT files")
    print(f"Setting sources to: {vrt_dir}\\[filename].vrt\n")
    
    updated_count = 0
    
    for vrt_file in vrt_files:
        try:
            was_updated = update_vrt_to_local_references(str(vrt_file))
            
            if was_updated:
                print(f"[OK] {vrt_file.name}")
                updated_count += 1
        
        except Exception as e:
            print(f"[ERROR] {vrt_file.name}: {str(e)}")
    
    print(f"\n[DONE] Updated {updated_count} / {len(vrt_files)} files\n")


if __name__ == "__main__":
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    
    # Update each VRT directory
    vrt_dirs = [
        os.path.join(base_dir, 'L5_VRT'),
        os.path.join(base_dir, 'L8_9_VRT-20251212T092651Z-1-001', 'L8_9_VRT'),
        os.path.join(base_dir, 'S2_VRT-20251212T092639Z-1-001', 'S2_VRT'),
        os.path.join(base_dir, 'Liss4Clip_VRT-20251212T092648Z-1-001', 'Liss4Clip_VRT')
    ]
    
    for vrt_dir in vrt_dirs:
        if os.path.exists(vrt_dir):
            batch_update_directory(vrt_dir)
        else:
            print(f"\n[SKIP] Directory not found: {vrt_dir}")
    
    print("="*70)
    print(" ALL UPDATES COMPLETE")
    print("="*70)
    print("\nVRT files now reference .vrt files in their own directories:")
    print(f"  L5_VRT/*.vrt -> F:\\NIDM-SRM Drone Project Docs\\L5_VRT\\*.vrt")
    print(f"  L8_9_VRT/*.vrt -> F:\\NIDM-SRM Drone Project Docs\\L8_9_VRT-...\\L8_9_VRT\\*.vrt")
    print(f"  etc.")





