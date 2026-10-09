"""
VRT Source Path Updater
Updates the source file paths in VRT files to point to correct locations
"""
import os
import re
from pathlib import Path


def find_source_files(base_dir, pattern="*.tif"):
    """
    Find all source TIFF files that match the pattern.
    
    Returns:
        dict: Mapping of filename to full path
    """
    source_files = {}
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.tif', '.tiff')):
                full_path = os.path.join(root, file)
                source_files[file] = full_path
    
    return source_files


def update_vrt_paths(vrt_file, new_base_path=None, dry_run=True):
    """
    Update the source file paths in a VRT file.
    
    Args:
        vrt_file (str): Path to the VRT file
        new_base_path (str): New base directory for source files
        dry_run (bool): If True, only show what would be changed
        
    Returns:
        bool: True if updated successfully
    """
    try:
        with open(vrt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all SourceFilename tags
        pattern = r'<SourceFilename[^>]*>(.*?)</SourceFilename>'
        matches = re.findall(pattern, content)
        
        if not matches:
            print(f"  No source filenames found in {os.path.basename(vrt_file)}")
            return False
        
        old_path = matches[0]
        print(f"\n  Current path: {old_path}")
        
        if new_base_path:
            # Extract just the filename
            filename = os.path.basename(old_path)
            new_path = os.path.join(new_base_path, filename)
            
            print(f"  New path:     {new_path}")
            
            if not dry_run:
                # Replace all occurrences
                new_content = content.replace(old_path, new_path)
                
                # Write back
                with open(vrt_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"  [OK] Updated!")
                return True
            else:
                print(f"  [DRY RUN] Would update to: {new_path}")
                return True
        else:
            print(f"  [INFO] No new path specified")
            return False
            
    except Exception as e:
        print(f"  [ERROR] Failed to update {vrt_file}: {e}")
        return False


def update_vrt_to_relative_paths(vrt_file, dry_run=True):
    """
    Update VRT to use relative paths instead of absolute paths.
    
    Args:
        vrt_file (str): Path to the VRT file
        dry_run (bool): If True, only show what would be changed
    """
    try:
        with open(vrt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find SourceFilename tags with relativeToVRT="0"
        pattern = r'<SourceFilename relativeToVRT="0">(.*?)</SourceFilename>'
        matches = re.findall(pattern, content)
        
        if not matches:
            return False
        
        vrt_dir = os.path.dirname(vrt_file)
        old_path = matches[0]
        filename = os.path.basename(old_path)
        
        # Create path relative to VRT location
        relative_path = filename  # Assuming source file is in same directory
        
        print(f"\n  VRT: {os.path.basename(vrt_file)}")
        print(f"  Old: {old_path}")
        print(f"  New: {relative_path} (relative)")
        
        if not dry_run:
            # Update to relative path
            new_content = content.replace(
                f'<SourceFilename relativeToVRT="0">{old_path}</SourceFilename>',
                f'<SourceFilename relativeToVRT="1">{relative_path}</SourceFilename>'
            )
            
            with open(vrt_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"  [OK] Updated to relative path!")
            return True
        else:
            print(f"  [DRY RUN] Would update to relative path")
            return True
            
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main():
    """Main function to update VRT files."""
    
    print("\n" + "="*70)
    print(" VRT SOURCE PATH UPDATER")
    print("="*70)
    
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    
    # VRT directories
    vrt_dirs = {
        'L5_VRT': os.path.join(base_dir, 'L5_VRT'),
        'L8_9_VRT': os.path.join(base_dir, 'L8_9_VRT-20251212T092651Z-1-001', 'L8_9_VRT'),
        'Liss4_VRT': os.path.join(base_dir, 'Liss4Clip_VRT-20251212T092648Z-1-001', 'Liss4Clip_VRT'),
        'S2_VRT': os.path.join(base_dir, 'S2_VRT-20251212T092639Z-1-001', 'S2_VRT')
    }
    
    # Ask user for action
    print("\nWhat would you like to do?")
    print("1. Show current paths in VRT files (analyze)")
    print("2. Update to new absolute path")
    print("3. Convert to relative paths")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        # Analyze mode
        print("\n" + "-"*70)
        print("ANALYZING VRT FILES")
        print("-"*70)
        
        for name, vrt_dir in vrt_dirs.items():
            if not os.path.exists(vrt_dir):
                continue
                
            print(f"\n### {name} ###")
            vrt_files = [f for f in os.listdir(vrt_dir) if f.endswith('.vrt')]
            
            if vrt_files:
                sample_vrt = os.path.join(vrt_dir, vrt_files[0])
                update_vrt_paths(sample_vrt, dry_run=True)
                print(f"  Total VRT files: {len(vrt_files)}")
    
    elif choice == '2':
        # Update to new path
        new_path = input("\nEnter new base directory path (where TIFF files are located): ").strip()
        
        if not os.path.exists(new_path):
            print(f"[ERROR] Directory not found: {new_path}")
            return
        
        dry_run = input("Dry run first? (y/n): ").strip().lower() == 'y'
        
        for name, vrt_dir in vrt_dirs.items():
            if not os.path.exists(vrt_dir):
                continue
            
            print(f"\n### Processing {name} ###")
            vrt_files = [f for f in os.listdir(vrt_dir) if f.endswith('.vrt')]
            
            for vrt_file in vrt_files:
                full_path = os.path.join(vrt_dir, vrt_file)
                update_vrt_paths(full_path, new_path, dry_run)
    
    elif choice == '3':
        # Convert to relative paths
        dry_run = input("Dry run first? (y/n): ").strip().lower() == 'y'
        
        for name, vrt_dir in vrt_dirs.items():
            if not os.path.exists(vrt_dir):
                continue
            
            print(f"\n### Processing {name} ###")
            vrt_files = [f for f in os.listdir(vrt_dir) if f.endswith('.vrt')]
            
            updated = 0
            for vrt_file in vrt_files:
                full_path = os.path.join(vrt_dir, vrt_file)
                if update_vrt_to_relative_paths(full_path, dry_run):
                    updated += 1
            
            print(f"\n  Updated: {updated}/{len(vrt_files)} files")
    
    print("\n" + "="*70)
    print(" COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()





