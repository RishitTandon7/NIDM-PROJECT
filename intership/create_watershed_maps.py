"""
Enhanced visualization with watershed boundaries overlay
Matching the professional GLOF hazard assessment map style
"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import fiona


def create_professional_map_with_watersheds(
    pred_map, 
    profile, 
    output_path, 
    title="GLOF Classification",
    watershed_shp=None
):
    """
    Create classification map with watershed boundaries overlay.
    
    Args:
        pred_map: 2D array of predictions
        profile: Rasterio profile with geotransform
        output_path: Where to save the map
        title: Map title
        watershed_shp: Path to watershed/catchment shapefile
    """
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Define colors - matching reference image style
    colors = ['#C8E6C9', '#64B5F6', '#FDD835', '#FF9800', '#D32F2F']  # 5 hazard levels
    class_names = ['Very Low', 'Low', 'Medium', 'High', 'Very Severe']
    
    # For 3-class glacier classification, use these:
    if np.max(pred_map) <= 2:
        colors = ['#90EE90', '#1E90FF', '#FF6347']
        class_names = ['Non-Glacier', 'Glacier', 'Glacier Lake']
    
    cmap = ListedColormap(colors)
    
    # Get geographic extent
    transform = profile['transform']
    left = transform[2]
    top = transform[5]
    right = left + profile['width'] * transform[0]
    bottom = top + profile['height'] * transform[4]
    extent = [left, right, bottom, top]
    
    # Plot classification
    ax.imshow(pred_map, cmap=cmap, extent=extent, aspect='auto', 
             interpolation='nearest', alpha=0.9)
    
    # Overlay watershed boundaries
    if watershed_shp:
        try:
            with fiona.open(watershed_shp) as shp:
                for feature in shp:
                    geom = feature['geometry']
                    
                    # Handle different geometry types
                    if geom['type'] == 'Polygon':
                        coords = geom['coordinates'][0]
                        x, y = zip(*coords)
                        ax.plot(x, y, color='black', linewidth=1.5, zorder=10)
                    
                    elif geom['type'] == 'MultiPolygon':
                        for polygon in geom['coordinates']:
                            coords = polygon[0]
                            x, y = zip(*coords)
                            ax.plot(x, y, color='black', linewidth=1.5, zorder=10)
                    
                    elif geom['type'] == 'LineString':
                        # For stream networks
                        coords = geom['coordinates']
                        x, y = zip(*coords)
                        ax.plot(x, y, color='darkblue', linewidth=0.8, 
                               alpha=0.7, zorder=9)
            
            print(f"  [OK] Watershed boundaries added")
        
        except Exception as e:
            print(f"  [!] Could not overlay watersheds: {str(e)[:50]}")
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=13, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    
    # Add grid
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5, color='gray')
    
    # Tick formatting
    ax.tick_params(labelsize=11)
    
    # Create legend
    legend_elements = [
        Patch(facecolor=colors[i], edgecolor='black', linewidth=0.5, label=class_names[i])
        for i in range(len(class_names))
    ]
    
    # Add watershed boundary to legend
    if watershed_shp:
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], color='black', linewidth=1.5, label='Watershed Boundary')
        )
    
    ax.legend(handles=legend_elements,
             loc='lower left',
             fontsize=11,
             frameon=True,
             fancybox=True,
             shadow=True,
             title='Susceptibility Zone' if len(class_names) == 5 else 'Classification',
             title_fontsize=12,
             edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# Example usage
if __name__ == "__main__":
    import rasterio
    import os
    from quick_train import load_shapefile_labels
    
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    
    # Load one of the generated classification maps to enhance
    map_file = os.path.join(base_dir, 'intership', 'maps', 
                           'Landsat_8_9_Landsat8_2014-11-07_classification.png')
    
    # Or generate a new one
    shp_path = os.path.join(base_dir, 'GLOF project', 'image analysis', 
                           'classified_glacier_dt.shp')
    img_path = os.path.join(base_dir, 'Data_Aligned', 'L8_image_aligned.tif')
    
    print("Creating professional map with watershed boundaries...")
    
    # Load image
    with rasterio.open(img_path) as src:
        # Downsample
        scale = min(500/src.width, 500/src.height)
        new_w = int(src.width * scale)
        new_h = int(src.height * scale)
        
        data = src.read(
            out_shape=(src.count, new_h, new_w),
            resampling=rasterio.enums.Resampling.bilinear
        )
        
        profile = src.profile.copy()
        profile.update({
            'width': new_w,
            'height': new_h,
            'transform': src.transform * src.transform.scale(
                (src.width / new_w),
                (src.height / new_h)
            )
        })
    
    # Load labels
    labels = load_shapefile_labels(shp_path, profile)
    
    if labels is not None:
        # Create enhanced map
        output_path = os.path.join(base_dir, 'intership', 
                                  'professional_watershed_map.png')
        
        create_professional_map_with_watersheds(
            labels,  # Use actual labels as the map
            profile,
            output_path,
            title='GLOF Hazard Zonation Map (Watershed-wise)',
            watershed_shp=shp_path
        )
        
        print(f"\n[OK] Professional map created: {output_path}")
    else:
        print("[ERROR] Could not load labels")





