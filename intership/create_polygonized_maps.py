"""
Create Polygonized Glacier Classification Maps
Matching the reference image style with smooth glacier polygons
"""
import os
import numpy as np
import rasterio
from rasterio.features import rasterize
import fiona
from scipy.ndimage import binary_opening, binary_closing, label
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches


def smooth_classification(pred_map, min_size=100, iterations=2):
    """
    Smooth classification map to remove scattered pixels.
    Creates smooth glacier polygons like reference image.
    """
    smoothed = pred_map.copy()
    
    # For each class, perform morphological operations
    for class_id in np.unique(pred_map):
        if class_id < 0:
            continue
        
        # Create binary mask for this class
        mask = (pred_map == class_id)
        
        # Remove small scattered pixels
        mask = remove_small_objects(mask, min_size=min_size)
        
        # Smooth boundaries with morphological operations
        mask = binary_closing(mask, iterations=iterations)
        mask = binary_opening(mask, iterations=iterations)
        
        # Update smoothed map
        smoothed[mask] = class_id
    
    return smoothed


def create_polygonized_map(shp_path, output_path, profile=None, title="Polygonized Classified Glacier Map"):
    """
    Create a smooth polygonized glacier map directly from shapefile labels.
    This matches the reference image style.
    
    Args:
        shp_path: Path to classified glacier shapefile
        output_path: Where to save the map
        profile: Optional rasterio profile for georeferencing
        title: Map title
    """
    
    # Get shapefile bounds
    with fiona.open(shp_path) as shp:
        bounds = shp.bounds
        features = list(shp)
    
    # Create raster profile if not provided
    if profile is None:
        # Create a reasonable resolution raster
        width = 600
        height = int(600 * (bounds[3] - bounds[1]) / (bounds[2] - bounds[0]))
        
        from rasterio.transform import from_bounds
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
        
        profile = {
            'driver': 'GTiff',
            'dtype': 'int16',
            'width': width,
            'height': height,
            'count': 1,
            'crs': 'EPSG:4326',
            'transform': transform
        }
    
    # Rasterize shapefile
    shapes = [
        (feature["geometry"], feature["properties"]["class_id"])
        for feature in features
    ]
    
    classification = rasterize(
        shapes=shapes,
        out_shape=(profile['height'], profile['width']),
        transform=profile['transform'],
        fill=-1,
        dtype='int32'
    )
    
    # Create smooth polygonized visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define colors matching reference image
    # Gray = Non-Glacier (0), Red = Glacier (1), Blue = Glacier Lake (2)
    colors = ['#808080', '#D55E5E', '#4A90E2']
    
    # Create colored image
    colored_image = np.zeros((profile['height'], profile['width'], 3))
    
    for class_id, color in enumerate(colors):
        mask = (classification == (class_id + 1))  # class_id in shapefile are 1,2,3
        rgb = [int(color[i:i+2], 16)/255 for i in (1, 3, 5)]
        for c in range(3):
            colored_image[:, :, c][mask] = rgb[c]
    
    # Get extent
    transform = profile['transform']
    left = transform[2]
    top = transform[5]
    right = left + profile['width'] * transform[0]
    bottom = top + profile['height'] * transform[4]
    extent = [left, right, bottom, top]
    
    # Display
    ax.imshow(colored_image, extent=extent, origin='upper', interpolation='bilinear')
    
    # Add watershed boundaries (polygons)
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
    except Exception as e:
        print(f"  [!] Could not add polygon boundaries: {str(e)[:50]}")
    
    # Formatting
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend - NO "Very Low, Low" labels
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='black', linewidth=0.5, label='Non-Glacier'),
        Patch(facecolor=colors[1], edgecolor='black', linewidth=0.5, label='Glacier'),
        Patch(facecolor=colors[2], edgecolor='black', linewidth=0.5, label='Glacier Lake')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', 
             fontsize=10, frameon=True, fancybox=False, shadow=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  [OK] Created polygonized map: {os.path.basename(output_path)}")


def main():
    """Create polygonized glacier maps for all datasets."""
    
    print("\n" + "="*70)
    print(" CREATING POLYGONIZED GLACIER MAPS")
    print(" (Matching Reference Image Style)")
    print("="*70 + "\n")
    
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    shp_path = os.path.join(base_dir, 'GLOF project', 'image analysis', 'classified_glacier_dt.shp')
    output_dir = os.path.join(base_dir, 'intership', 'polygonized_maps')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create maps for different satellite types
    datasets = {
        'Sentinel2_2021-12': 'Sentinel-2 December 2021',
        'Landsat8_2021': 'Landsat 8 2021',
        'Landsat5_1988': 'Landsat 5 1988'
    }
    
    for filename, title in datasets.items():
        output_path = os.path.join(output_dir, f'{filename}_polygonized.png')
        
        print(f"\nCreating: {title}")
        create_polygonized_map(
            shp_path=shp_path,
            output_path=output_path,
            title=f'Polygonized Classified Glacier Map\n{title}'
        )
    
    print("\n" + "="*70)
    print(" COMPLETE")
    print("="*70)
    print(f"\nMaps saved to: {output_dir}")
    print("\nNOTE: These maps show the GROUND TRUTH labels from your shapefile")
    print("      in a smooth, polygonized style matching your reference image.")


if __name__ == "__main__":
    main()





