"""
Batch Create Polygonized Glacier Maps for ALL VRT Files
Processes all dates from L5, L8/9, S2, and LISS-IV
"""
import os
import numpy as np
import rasterio
from rasterio.features import rasterize
import fiona
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path


def create_polygonized_map_fast(shp_path, output_path, date_str, sensor_name):
    """
    Create a smooth polygonized glacier map.
    Fast version for batch processing.
    """
    
    # Get shapefile bounds
    with fiona.open(shp_path) as shp:
        bounds = shp.bounds
        features = list(shp)
    
    # Create raster profile
    width = 800
    height = int(800 * (bounds[3] - bounds[1]) / (bounds[2] - bounds[0]))
    
    from rasterio.transform import from_bounds
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    
    # Rasterize shapefile
    shapes = [
        (feature["geometry"], feature["properties"]["class_id"])
        for feature in features
    ]
    
    classification = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=-1,
        dtype='int32'
    )
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Colors: Gray, Red, Blue
    colors = ['#808080', '#D55E5E', '#4A90E2']
    
    # Create colored image
    colored_image = np.zeros((height, width, 3))
    
    for class_id, color in enumerate(colors):
        mask = (classification == (class_id + 1))
        rgb = [int(color[i:i+2], 16)/255 for i in (1, 3, 5)]
        for c in range(3):
            colored_image[:, :, c][mask] = rgb[c]
    
    # Get extent
    left = transform[2]
    top = transform[5]
    right = left + width * transform[0]
    bottom = top + height * transform[4]
    extent = [left, right, bottom, top]
    
    # Display
    ax.imshow(colored_image, extent=extent, origin='upper', interpolation='bilinear')
    
    # Add polygon boundaries
    try:
        with fiona.open(shp_path) as shp:
            for feature in shp:
                geom = feature['geometry']
                
                if geom['type'] == 'Polygon':
                    coords = geom['coordinates'][0]
                    x, y = zip(*coords)
                    ax.plot(x, y, color='black', linewidth=0.5, alpha=0.7)
                
                elif geom['type'] == 'MultiPolygon':
                    for polygon in geom['coordinates']:
                        coords = polygon[0]
                        x, y = zip(*coords)
                        ax.plot(x, y, color='black', linewidth=0.5, alpha=0.7)
    except:
        pass
    
    # Formatting
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title(f'Polygonized Classified Glacier Map\n{sensor_name} - {date_str}', 
                fontsize=13, fontweight='bold')
    
    # Legend - CORRECTED MAPPING
    # class_id 1 = Glacier, 2 = Glacier Lake, 3 = Non-Glacier
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='black', linewidth=0.5, label='Glacier'),
        Patch(facecolor=colors[1], edgecolor='black', linewidth=0.5, label='Glacier Lake'),
        Patch(facecolor=colors[2], edgecolor='black', linewidth=0.5, label='Non-Glacier')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', 
             fontsize=9, frameon=True, fancybox=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def process_vrt_directory(vrt_dir, sensor_name, shp_path, output_base_dir):
    """Process all VRT files in a directory."""
    
    print(f"\n{'='*70}")
    print(f" Processing {sensor_name}")
    print(f"{'='*70}\n")
    
    # Get all VRT files
    vrt_files = sorted(list(Path(vrt_dir).glob('*.vrt')))
    
    if not vrt_files:
        print("  No VRT files found!")
        return 0
    
    print(f"  Found {len(vrt_files)} VRT files")
    
    # Create output directory for this sensor
    output_dir = os.path.join(output_base_dir, sensor_name.replace(' ', '_').replace('/', '_'))
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    
    for idx, vrt_file in enumerate(vrt_files, 1):
        # Extract date from filename
        date_str = vrt_file.stem.replace('_bands_clipped', '').replace('_clipped', '').replace('_bands', '')
        
        # Create output filename
        output_filename = f'{date_str}_polygonized.png'
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            create_polygonized_map_fast(shp_path, output_path, date_str, sensor_name)
            print(f"  [{idx}/{len(vrt_files)}] {date_str}")
            count += 1
        
        except Exception as e:
            print(f"  [ERROR] {date_str}: {str(e)[:50]}")
    
    print(f"\n  [DONE] Created {count} / {len(vrt_files)} maps")
    return count


def main():
    """Process all VRT directories."""
    
    print("\n" + "="*70)
    print(" BATCH POLYGONIZED GLACIER MAP GENERATION")
    print(" Creating maps for ALL VRT files")
    print("="*70 + "\n")
    
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    shp_path = os.path.join(base_dir, 'GLOF project', 'image analysis', 'classified_glacier_dt.shp')
    output_base = os.path.join(base_dir, 'intership', 'all_polygonized_maps')
    
    # VRT directories to process
    vrt_configs = [
        {
            'dir': os.path.join(base_dir, 'L5_VRT'),
            'name': 'Landsat_5'
        },
        {
            'dir': os.path.join(base_dir, 'L8_9_VRT-20251212T092651Z-1-001', 'L8_9_VRT'),
            'name': 'Landsat_8/9'
        },
        {
            'dir': os.path.join(base_dir, 'S2_VRT-20251212T092639Z-1-001', 'S2_VRT'),
            'name': 'Sentinel_2'
        },
        {
            'dir': os.path.join(base_dir, 'Liss4Clip_VRT-20251212T092648Z-1-001', 'Liss4Clip_VRT'),
            'name': 'LISS_IV'
        }
    ]
    
    total_maps = 0
    
    for config in vrt_configs:
        if os.path.exists(config['dir']):
            count = process_vrt_directory(
                config['dir'],
                config['name'],
                shp_path,
                output_base
            )
            total_maps += count
        else:
            print(f"\n[SKIP] Directory not found: {config['dir']}")
    
    print("\n" + "="*70)
    print(f" BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"\nTotal maps created: {total_maps}")
    print(f"Output directory: {output_base}")
    print("\nMaps organized by sensor:")
    print("  - Landsat_5/")
    print("  - Landsat_8_9/")
    print("  - Sentinel_2/")
    print("  - LISS_IV/")


if __name__ == "__main__":
    main()





