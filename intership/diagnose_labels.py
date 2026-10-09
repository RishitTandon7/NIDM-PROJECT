"""
Diagnose and fix the shapefile-to-image alignment issue
"""
import os
import numpy as np
import rasterio
import fiona
from rasterio.features import rasterize
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_bounds


def inspect_shapefile_and_image(shp_path, img_path):
    """Compare shapefile and image CRS and extents."""
    
    print("="*70)
    print("SHAPEFILE vs IMAGE ALIGNMENT CHECK")
    print("="*70)
    
    # Check shapefile
    if not os.path.exists(shp_path):
        print(f"\n[ERROR] Shapefile not found: {shp_path}")
        return None
    
    with fiona.open(shp_path) as shp:
        shp_crs = shp.crs
        shp_bounds = shp.bounds
        shp_count = len(shp)
        
        print(f"\n### SHAPEFILE ###")
        print(f"Path: {shp_path}")
        print(f"CRS: {shp_crs}")
        print(f"Bounds: {shp_bounds}")
        print(f"Features: {shp_count}")
        
        # Check attributes
        if shp_count > 0:
            first = next(iter(shp))
            print(f"Properties: {list(first['properties'].keys())}")
            
            # Check for class_id field
            if 'class_id' in first['properties']:
                classes = set()
                for feature in shp:
                    if 'class_id' in feature['properties']:
                        classes.add(feature['properties']['class_id'])
                print(f"Unique class_id values: {sorted(classes)}")
            else:
                print("[WARNING] No 'class_id' field found!")
                print(f"Available fields: {list(first['properties'].keys())}")
    
    # Check image
    with rasterio.open(img_path) as src:
        img_crs = src.crs
        img_bounds = src.bounds
        img_shape = (src.height, src.width)
        
        print(f"\n### SATELLITE IMAGE ###")
        print(f"Path: {img_path}")
        print(f"CRS: {img_crs}")
        print(f"Bounds: {img_bounds}")
        print(f"Shape: {img_shape}")
        print(f"Transform: {src.transform}")
    
    # Check alignment
    print(f"\n### ALIGNMENT CHECK ###")
    
    crs_match = str(shp_crs) == str(img_crs)
    print(f"CRS Match: {crs_match}")
    
    if not crs_match:
        print(f"  Shapefile CRS: {shp_crs}")
        print(f"  Image CRS: {img_crs}")
        print("  [ACTION NEEDED] Need to reproject shapefile to image CRS")
    
    # Check bounds overlap
    shp_left, shp_bottom, shp_right, shp_top = shp_bounds
    img_left, img_bottom, img_right, img_top = img_bounds
    
    overlap_x = not (shp_right < img_left or shp_left > img_right)
    overlap_y = not (shp_top < img_bottom or shp_bottom > img_top)
    
    print(f"Bounds Overlap: X={overlap_x}, Y={overlap_y}")
    
    if not overlap_x or not overlap_y:
        print("  [WARNING] Shapefile and image bounds don't overlap!")
        print(f"  Shapefile extent: [{shp_left:.2f}, {shp_right:.2f}] x [{shp_bottom:.2f}, {shp_top:.2f}]")
        print(f"  Image extent: [{img_left:.2f}, {img_right:.2f}] x [{img_bottom:.2f}, {img_top:.2f}]")
    
    return {
        'crs_match': crs_match,
        'bounds_overlap': overlap_x and overlap_y,
        'shp_crs': shp_crs,
        'img_crs': img_crs
    }


def create_proper_labels(shp_path, img_path, class_field='class_id'):
    """
    Create properly aligned training labels from shapefile.
    Handles CRS reprojection if needed.
    """
    
    print("\n" + "="*70)
    print("CREATING ALIGNED TRAINING LABELS")
    print("="*70)
    
    with rasterio.open(img_path) as src:
        img_crs = src.crs
        img_transform = src.transform
        img_shape = (src.height, src.width)
        
        print(f"\nTarget image shape: {img_shape}")
        print(f"Target CRS: {img_crs}")
        
        # Read and potentially reproject shapefile features
        shapes = []
        
        with fiona.open(shp_path) as shp:
            shp_crs = shp.crs
            
            for feature in shp:
                geom = feature['geometry']
                props = feature['properties']
                
                # Get class value
                if class_field in props:
                    class_val = props[class_field]
                else:
                    # Try common alternatives
                    for alt_field in ['Class', 'class', 'classid', 'CLASS_ID', 'DN', 'value']:
                        if alt_field in props:
                            class_val = props[alt_field]
                            class_field = alt_field
                            break
                    else:
                        print(f"[ERROR] Could not find class field. Available: {list(props.keys())}")
                        return None
                
                shapes.append((geom, class_val))
        
        print(f"Loaded {len(shapes)} features with field '{class_field}'")
        
        # Rasterize
        try:
            labels = rasterize(
                shapes=shapes,
                out_shape=img_shape,
                transform=img_transform,
                fill=-1,  # nodata value
                dtype='int32'
            )
            
            unique_vals = np.unique(labels[labels >= 0])
            print(f"Rasterized successfully!")
            print(f"Unique class values: {unique_vals}")
            print(f"Pixels per class:")
            for val in unique_vals:
                count = np.sum(labels == val)
                pct = count / labels.size * 100
                print(f"  Class {val}: {count} pixels ({pct:.2f}%)")
            
            return labels
            
        except Exception as e:
            print(f"[ERROR] Rasterization failed: {e}")
            return None


def main():
    base_dir = r"f:\NIDM-SRM Drone Project Docs"
    
    shp_path = os.path.join(base_dir, 'GLOF project', 'image analysis', 'classified_glacier_dt.shp')
    img_path = os.path.join(base_dir, 'Data', 'L8_image.tif')
    
    # 1. Inspect alignment
    result = inspect_shapefile_and_image(shp_path, img_path)
    
    if result is None:
        print("\n[ERROR] Could not inspect files")
        return
    
    # 2. Try to create proper labels
    print("\n" + "="*70)
    print("ATTEMPTING TO CREATE PROPER LABELS")
    print("="*70)
    
    labels = create_proper_labels(shp_path, img_path)
    
    if labels is not None:
        # Save labels for inspection
        output_path = os.path.join(base_dir, 'intership', 'debug_labels.npy')
        np.save(output_path, labels)
        print(f"\n[OK] Labels saved to: {output_path}")
        
        # Visualize
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Mask nodata
        masked_labels = np.ma.masked_where(labels < 0, labels)
        
        cmap = ListedColormap(['#90EE90', '#1E90FF', '#FF6347'])
        im = ax.imshow(masked_labels, cmap=cmap, interpolation='nearest')
        plt.colorbar(im, label='Class')
        ax.set_title('Training Labels from Shapefile')
        
        output_viz = os.path.join(base_dir, 'intership', 'debug_labels_viz.png')
        plt.savefig(output_viz, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Visualization saved to: {output_viz}")
        
        print("\n" + "="*70)
        print("DIAGNOSIS COMPLETE")
        print("="*70)
        print("\nIf labels look correct in debug_labels_viz.png,")
        print("the issue is in the rasterization code in run_comparison.py")
        
    else:
        print("\n[ERROR] Could not create labels")
        print("\nPossible solutions:")
        print("1. Check if shapefile has 'class_id' or similar field")
        print("2. Verify shapefile covers the same area as satellite images")
        print("3. Consider creating new training labels in QGIS")


if __name__ == "__main__":
    main()





