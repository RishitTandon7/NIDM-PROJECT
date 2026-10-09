"""
Professional Watershed Classification Map Generator
Creates classification maps with coordinate axes and legends
Similar to professional GLOF hazard assessment maps
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import rasterio
from rasterio.plot import show
import fiona


def create_professional_classification_map(
    classification_array,
    profile,
    output_path,
    title="Classification Map",
    class_names=None,
    class_colors=None,
    watershed_shp=None,
    streams_shp=None
):
    """
    Create a professional classification map with watershed boundaries.
    
    Args:
        classification_array (np.ndarray): 2D array of classifications
        profile (dict): Rasterio profile with geotransform info
        output_path (str): Path to save the map
        title (str): Map title
        class_names (list): Names for each class
        class_colors (list): Colors for each class
        watershed_shp (str): Path to watershed boundary shapefile
        streams_shp (str): Path to stream network shapefile
    """
    # Default class setup
    if class_names is None:
        class_names = ['Non-Glacier', 'Glacier', 'Glacier Lake']
    
    if class_colors is None:
        class_colors = ['#90EE90', '#1E90FF', '#FF4500']  # light green, blue, red-orange
    
    # Create figure with proper size
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get geographic extent from profile
    transform = profile['transform']
    height, width = profile['height'], profile['width']
    
    # Calculate coordinates
    left = transform[2]
    top = transform[5]
    right = left + width * transform[0]
    bottom = top + height * transform[4]
    
    extent = [left, right, bottom, top]
    
    # Create colormap
    n_classes = len(class_names)
    cmap = ListedColormap(class_colors[:n_classes])
    bounds = np.arange(n_classes + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Plot classification
    im = ax.imshow(classification_array, 
                   cmap=cmap, 
                   norm=norm,
                   extent=extent,
                   interpolation='nearest',
                   aspect='auto')
    
    # Add watershed boundaries if available
    if watershed_shp and os.path.exists(watershed_shp):
        try:
            with fiona.open(watershed_shp) as shp:
                for feature in shp:
                    geom = feature['geometry']
                    if geom['type'] == 'Polygon':
                        coords = geom['coordinates'][0]
                        x, y = zip(*coords)
                        ax.plot(x, y, color='black', linewidth=1.5, zorder=10)
                    elif geom['type'] == 'MultiPolygon':
                        for polygon in geom['coordinates']:
                            coords = polygon[0]
                            x, y = zip(*coords)
                            ax.plot(x, y, color='black', linewidth=1.5, zorder=10)
        except Exception as e:
            print(f"Could not plot watersheds: {e}")
    
    # Stream network would require similar handling, skip for now
    if streams_shp:
        pass  # Skip streams for simplicity
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Create legend
    legend_elements = [
        Patch(facecolor=class_colors[i], edgecolor='black', label=class_names[i])
        for i in range(n_classes)
    ]
    
    ax.legend(handles=legend_elements,
             loc='lower left',
             fontsize=10,
             frameon=True,
             fancybox=True,
             shadow=True,
             title='Classification',
             title_fontsize=11)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[OK] Professional map saved to: {os.path.basename(output_path)}")


def create_multi_class_hazard_map(
    classification_array,
    profile,
    output_path,
    title="GLOF Hazard Zonation Map",
    hazard_levels=5,
    watershed_shp=None
):
    """
    Create a multi-class hazard zonation map.
    
    Args:
        classification_array (np.ndarray): 2D array of classifications
        profile (dict): Rasterio profile
        output_path (str): Output path
        title (str): Map title
        hazard_levels (int): Number of hazard zones
        watershed_shp (str): Watershed boundary shapefile
    """
    # Define hazard color scheme (from low to very high)
    if hazard_levels == 5:
        colors = ['#90EE90', '#87CEEB', '#FFD700', '#FFA500', '#FF4500']
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    elif hazard_levels == 4:
        colors = ['#90EE90', '#FFD700', '#FFA500', '#FF4500']
        labels = ['Low', 'Medium', 'High', 'Very Severe']
    else:
        colors = ['#90EE90', '#1E90FF', '#FF4500']
        labels = ['Low', 'Medium', 'High']
    
    create_professional_classification_map(
        classification_array=classification_array,
        profile=profile,
        output_path=output_path,
        title=title,
        class_names=labels,
        class_colors=colors,
        watershed_shp=watershed_shp
    )


def create_index_classification_map(
    index_array,
    profile,
    output_path,
    index_name="Index",
    thresholds=None,
    watershed_shp=None
):
    """
    Create a classification map based on index thresholds.
    
    Args:
        index_array (np.ndarray): 2D array of index values
        profile (dict): Rasterio profile
        output_path (str): Output path
        index_name (str): Name of the index
        thresholds (list): Threshold values for classification
        watershed_shp (str): Watershed boundary shapefile
    """
    if thresholds is None:
        # Default 5-class thresholds
        thresholds = [0.16, 0.31, 0.47, 0.63]
    
    # Classify based on thresholds
    classified = np.zeros_like(index_array, dtype=int)
    for i, threshold in enumerate(thresholds):
        classified[index_array >= threshold] = i + 1
    
    # Define labels and colors
    n_classes = len(thresholds) + 1
    if n_classes == 5:
        labels = [
            f'Very Low (< {thresholds[0]:.2f})',
            f'Low ({thresholds[0]:.2f} - {thresholds[1]:.2f})',
            f'Medium ({thresholds[1]:.2f} - {thresholds[2]:.2f})',
            f'High ({thresholds[2]:.2f} - {thresholds[3]:.2f})',
            f'Very High (≥ {thresholds[3]:.2f})'
        ]
        colors = ['#90EE90', '#87CEEB', '#FFD700', '#FFA500', '#FF4500']
    else:
        labels = [f'Class {i}' for i in range(n_classes)]
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, n_classes))
    
    create_professional_classification_map(
        classification_array=classified,
        profile=profile,
        output_path=output_path,
        title=f"Watershed-wise {index_name} Classification",
        class_names=labels,
        class_colors=colors,
        watershed_shp=watershed_shp
    )


def main():
    """Demonstrate professional map creation."""
    
    print("\n" + "="*70)
    print(" PROFESSIONAL CLASSIFICATION MAP GENERATOR")
    print("="*70 + "\n")
    
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    
    # Look for watershed shapefile
    watershed_shp = os.path.join(base_dir, 'GLOF project', 'image analysis', 'classified_glacier_dt.shp')
    if not os.path.exists(watershed_shp):
        watershed_shp = None
        print("[INFO] Watershed shapefile not found, maps will be created without boundaries")
    else:
        print(f"[OK] Using watershed shapefile: {os.path.basename(watershed_shp)}")
    
    # Load satellite data
    data_dir = os.path.join(base_dir, 'Data')
    output_dir = os.path.join(base_dir, 'intership')
    
    # Example: Load and classify Landsat 8 data
    l8_path = os.path.join(data_dir, 'L8_image.tif')
    
    if os.path.exists(l8_path):
        print(f"\n[INFO] Processing: {os.path.basename(l8_path)}")
        
        with rasterio.open(l8_path) as src:
            # Read data
            data = src.read()
            profile = src.profile
            
            # Calculate NDVI for demonstration
            if data.shape[0] >= 4:
                nir = data[3].astype(float)
                red = data[2].astype(float)
                
                ndvi = np.divide(nir - red, nir + red, 
                               out=np.zeros_like(nir), 
                               where=(nir + red) != 0)
                
                # Create NDVI classification map
                print("[INFO] Creating NDVI classification map...")
                create_index_classification_map(
                    index_array=ndvi,
                    profile=profile,
                    output_path=os.path.join(output_dir, 'professional_ndvi_classification.png'),
                    index_name='NDVI',
                    thresholds=[0.2, 0.3, 0.4, 0.5],
                    watershed_shp=watershed_shp
                )
            
            # Create a simple glacier classification map
            # For demonstration, create synthetic classification
            height, width = data.shape[1], data.shape[2]
            classification = np.random.randint(0, 3, size=(height, width))
            
            print("[INFO] Creating glacier classification map...")
            create_professional_classification_map(
                classification_array=classification,
                profile=profile,
                output_path=os.path.join(output_dir, 'professional_glacier_classification.png'),
                title='Watershed-wise Glacier Classification',
                class_names=['Non-Glacier', 'Glacier', 'Glacier Lake'],
                class_colors=['#90EE90', '#1E90FF', '#FF4500'],
                watershed_shp=watershed_shp
            )
            
            # Create hazard zonation map
            print("[INFO] Creating hazard zonation map...")
            hazard_zones = np.random.randint(0, 5, size=(height, width))
            create_multi_class_hazard_map(
                classification_array=hazard_zones,
                profile=profile,
                output_path=os.path.join(output_dir, 'professional_hazard_zonation.png'),
                title='GLOF Hazard Zonation Map',
                hazard_levels=5,
                watershed_shp=watershed_shp
            )
    
    print("\n" + "="*70)
    print(" COMPLETE")
    print("="*70 + "\n")
    print("Generated professional classification maps with:")
    print("  - Watershed boundaries (if shapefile provided)")
    print("  - Latitude/Longitude coordinates")
    print("  - Professional legends")
    print("  - Multi-class color schemes")


if __name__ == "__main__":
    main()





