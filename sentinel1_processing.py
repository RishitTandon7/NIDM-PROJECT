"""
Sentinel-1 SAR Processing — 3-Class Classifier
=================================================
Classes:
  0 = Non-Glacier  (terrain, rock, shadow)
  1 = Glacier      (snow/ice)
  2 = Glacier Lake (proglacial / supraglacial lakes)

Two processing modes:
  A) Google Earth Engine  — cloud processing, exports GeoTiff + GeoJSON
  B) Local (rasterio)     — offline processing on downloaded GeoTIFFs

Author: Antigravity
"""

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

# ─────────────────────────────────────────────────────────────
# LABEL MAP  (shared by both GEE and local paths)
# ─────────────────────────────────────────────────────────────
CLASS_LABELS = {
    0: "Non-Glacier",
    1: "Glacier",
    2: "Glacier Lake",
}
CLASS_COLORS = {
    0: "#8B6914",   # brown  – terrain/rock
    1: "#E8F4FD",   # icy white  – glacier
    2: "#1A7FD4",   # blue   – lake
}


# ═══════════════════════════════════════════════════════════════
# MODE A ─ GOOGLE EARTH ENGINE
# ═══════════════════════════════════════════════════════════════

def process_sentinel1_gee(longitude, latitude,
                           start_date='2023-06-01', end_date='2023-09-30'):
    """
    Build a Sentinel-1 median composite + feature stack for the ROI.

    Features added:
      VV, VH, VH/VV ratio, slope (terrain), elevation (SRTM)

    Returns
    -------
    input_img : ee.Image   – multi-band feature image
    roi       : ee.Geometry
    """
    import ee

    try:
        ee.Initialize()
    except Exception as e:
        print("Earth Engine not initialised. Run ee.Authenticate() first.")
        raise e

    roi = ee.Geometry.Point([longitude, latitude]).buffer(10000)

    # Sentinel-1 median composite (IW mode, all passes)
    s1_col = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterBounds(roi)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.eq('instrumentMode', 'IW')))

    composite = s1_col.median().clip(roi)

    # Feature engineering
    ratio     = composite.select('VH').divide(composite.select('VV')).rename('VH_VV_ratio')
    dem       = ee.Image('USGS/SRTMGL1_003').clip(roi)
    slope     = ee.Terrain.slope(dem).rename('slope')
    elevation = dem.rename('elevation')

    input_img = composite.addBands([ratio, slope, elevation])

    print("Sentinel-1 composite ready.")
    print(f"Feature bands: VV, VH, VH_VV_ratio, slope, elevation")
    return input_img, roi


def train_and_classify_gee(input_img, training_pts):
    """
    3-class supervised classification on GEE.

    Parameters
    ----------
    input_img    : ee.Image          – output of process_sentinel1_gee()
    training_pts : ee.FeatureCollection
                    Must have a 'label' property:
                      0 = Non-Glacier | 1 = Glacier | 2 = Glacier Lake

    Returns
    -------
    classified : ee.Image  (pixel values 0 / 1 / 2)
    """
    import ee

    FEATURE_BANDS = ['VV', 'VH', 'VH_VV_ratio', 'slope', 'elevation']

    training = input_img.sampleRegions(
        collection=training_pts,
        properties=['label'],
        scale=10
    )

    classifier = (ee.Classifier.smileRandomForest(100)
                  .train(features=training,
                         classProperty='label',
                         inputProperties=FEATURE_BANDS))

    classified = input_img.classify(classifier)

    # Remap so class names are explicit  (0/1/2 kept as-is)
    print("3-class GEE classification complete.")
    print("  0 = Non-Glacier  |  1 = Glacier  |  2 = Glacier Lake")
    return classified


def export_results_gee(classified, roi,
                        prefix='S1_3class',
                        folder='GEE_Exports'):
    """
    Export classified image to Google Drive as GeoTiff + GeoJSON.
    Each class polygon is exported as a separate GeoJSON file.
    """
    import ee

    # ── Raster export ──
    task_raster = ee.batch.Export.image.toDrive(
        image=classified.toByte(),
        description=f'{prefix}_raster',
        folder=folder,
        scale=10,
        region=roi,
        crs='EPSG:4326',
        maxPixels=1e9
    )
    task_raster.start()
    print(f"[GEE] Raster export started → Drive/{folder}/{prefix}_raster.tif")

    # ── Vector export — one GeoJSON per class ──
    tasks_vector = {}
    for class_val, class_name in CLASS_LABELS.items():
        binary = classified.eq(class_val).selfMask()
        vectors = binary.reduceToVectors(
            scale=10,
            geometry=roi,
            geometryType='polygon',
            eightConnected=True,
            labelProperty='class'
        ).map(lambda f: f.set('class_name', class_name).set('class_id', class_val))

        task = ee.batch.Export.table.toDrive(
            collection=vectors,
            description=f'{prefix}_{class_name.replace(" ", "_")}',
            folder=folder,
            fileFormat='GeoJSON'
        )
        task.start()
        tasks_vector[class_name] = task
        print(f"[GEE] Vector export started → Drive/{folder}/{prefix}_{class_name}.geojson")

    return task_raster, tasks_vector


# ═══════════════════════════════════════════════════════════════
# MODE B ─ LOCAL PROCESSING  (rasterio + scikit-learn)
# ═══════════════════════════════════════════════════════════════

def build_feature_stack_local(vv_path, vh_path, slope_path=None, elevation_path=None):
    """
    Stack SAR bands into a (H, W, n_features) array.

    Minimum required: vv_path, vh_path
    Optional:         slope_path, elevation_path

    Features: VV, VH, VH/VV ratio [, slope] [, elevation]
    """
    with rasterio.open(vv_path) as src:
        vv      = src.read(1).astype(np.float32)
        profile = src.profile

    with rasterio.open(vh_path) as src:
        vh = src.read(1).astype(np.float32)

    # Avoid divide-by-zero
    ratio = np.where(vv != 0, vh / vv, 0).astype(np.float32)

    bands = [vv, vh, ratio]
    names = ['VV', 'VH', 'VH_VV_ratio']

    if slope_path:
        with rasterio.open(slope_path) as src:
            bands.append(src.read(1).astype(np.float32))
        names.append('slope')

    if elevation_path:
        with rasterio.open(elevation_path) as src:
            bands.append(src.read(1).astype(np.float32))
        names.append('elevation')

    stack = np.dstack(bands)   # (H, W, n_features)
    print(f"Feature stack shape: {stack.shape}  →  bands: {names}")
    return stack, profile, names


def train_classifier_local(stack, labels_2d, n_estimators=200, test_size=0.3):
    """
    Train a 3-class Random Forest.

    Parameters
    ----------
    stack    : ndarray (H, W, n_features)
    labels_2d: ndarray (H, W) with values 0 / 1 / 2
                0=Non-Glacier | 1=Glacier | 2=Glacier Lake

    Returns
    -------
    clf        : fitted RandomForestClassifier
    valid_mask : (H*W,) bool – pixels used for training
    report     : str – sklearn classification report
    """
    H, W, F = stack.shape
    X = stack.reshape(-1, F)
    y = labels_2d.ravel()

    # Drop unlabelled pixels (nodata = 255 or -1)
    valid = (y >= 0) & (y <= 2) & ~np.isnan(X).any(axis=1)
    X_v, y_v = X[valid], y[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X_v, y_v, test_size=test_size, random_state=42, stratify=y_v
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',   # handles imbalanced glaciers vs terrain
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=[CLASS_LABELS[i] for i in sorted(CLASS_LABELS)]
    )
    print("\n── 3-Class Classification Report ──")
    print(report)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return clf, valid, report


def classify_image_local(clf, stack):
    """
    Run prediction over the full image stack.

    Returns class_map (H, W) with values 0 / 1 / 2.
    """
    H, W, F = stack.shape
    X = stack.reshape(-1, F)

    # Predict in chunks to keep memory manageable
    chunk = 100_000
    y_pred = np.empty(H * W, dtype=np.uint8)
    for i in range(0, H * W, chunk):
        y_pred[i:i+chunk] = clf.predict(X[i:i+chunk])

    class_map = y_pred.reshape(H, W)

    # Print area breakdown
    total = H * W
    for val, name in CLASS_LABELS.items():
        pct = np.sum(class_map == val) / total * 100
        print(f"  {name:15s}: {pct:.2f}%")

    return class_map


def save_results_local(class_map, profile, output_dir, date_str,
                        vv_path=None):
    """
    Save:
      • PNG visualisation (RGB composite + classified map + overlay)
      • GeoTiff  – single-band uint8
      • GeoJSON  – one file per class (Glacier, Glacier Lake)
    """
    os.makedirs(output_dir, exist_ok=True)
    H, W = class_map.shape

    # ── Visualisation ──────────────────────────────────────────
    cmap  = ListedColormap([CLASS_COLORS[i] for i in sorted(CLASS_COLORS)])
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Class map
    axes[0].imshow(class_map, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
    legend = [Patch(facecolor=CLASS_COLORS[i], edgecolor='k',
                    label=CLASS_LABELS[i]) for i in sorted(CLASS_LABELS)]
    axes[0].legend(handles=legend, loc='lower right', fontsize=10)
    axes[0].set_title(f'3-Class SAR Classification — {date_str}', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # VV backscatter thumbnail (if available)
    if vv_path and os.path.exists(vv_path):
        with rasterio.open(vv_path) as src:
            vv = src.read(1).astype(np.float32)
        axes[1].imshow(vv, cmap='gray', vmin=np.nanpercentile(vv, 2),
                       vmax=np.nanpercentile(vv, 98))
        axes[1].set_title('Sentinel-1 VV Backscatter', fontsize=13)
    else:
        axes[1].axis('off')
    axes[1].axis('off')

    plt.tight_layout()
    png_path = os.path.join(output_dir, f'S1_Classification_{date_str}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualisation saved: {png_path}")

    # ── GeoTiff ────────────────────────────────────────────────
    tif_prof = profile.copy()
    tif_prof.update({'dtype': rasterio.uint8, 'count': 1, 'compress': 'lzw'})
    tif_path = os.path.join(output_dir, f'S1_Classification_{date_str}.tif')
    with rasterio.open(tif_path, 'w', **tif_prof) as dst:
        dst.write(class_map, 1)
        # Write class names as metadata tags
        dst.update_tags(
            CLASS_0='Non-Glacier',
            CLASS_1='Glacier',
            CLASS_2='Glacier Lake'
        )
    print(f"✓ GeoTiff saved:       {tif_path}")

    # ── GeoJSON — one per class ─────────────────────────────────
    for class_val, class_name in CLASS_LABELS.items():
        binary = (class_map == class_val).astype(np.uint8)
        if not np.any(binary):
            print(f"  ⚠  No pixels for class '{class_name}', skipping GeoJSON.")
            continue
        polys = [
            {'geometry': geom, 'properties': {'class_id': class_val,
                                               'class_name': class_name,
                                               'date': date_str}}
            for geom, val in shapes(binary, mask=binary.astype(bool),
                                    transform=profile['transform'])
            if val == 1
        ]
        if polys:
            gdf = gpd.GeoDataFrame.from_features(polys, crs=profile['crs'])
            fname = f'S1_{class_name.replace(" ", "_")}_{date_str}.geojson'
            vec_path = os.path.join(output_dir, fname)
            gdf.to_file(vec_path, driver='GeoJSON')
            print(f"✓ GeoJSON saved:       {vec_path}  ({len(gdf)} polygons)")

    return tif_path


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE RUNNER — local mode
# ═══════════════════════════════════════════════════════════════

def run_local(vv_path, vh_path, labels_2d, output_dir,
              slope_path=None, elevation_path=None, date_str='YYYYMMDD'):
    """
    One-call pipeline for local Sentinel-1 3-class classification.

    Parameters
    ----------
    vv_path    : str – path to VV backscatter GeoTiff (terrain-corrected)
    vh_path    : str – path to VH backscatter GeoTiff
    labels_2d  : ndarray (H, W) with training labels:
                   0 = Non-Glacier | 1 = Glacier | 2 = Glacier Lake
                   Set unlabelled pixels to 255
    output_dir : str – folder for results
    slope_path : str (optional) – slope GeoTiff
    elevation_path : str (optional) – DEM GeoTiff
    date_str   : str – appended to output filenames
    """
    print(f"\n{'='*60}")
    print("Sentinel-1  ·  3-Class Glacier Classifier  ·  Local Mode")
    print(f"{'='*60}\n")

    # Build features
    stack, profile, feat_names = build_feature_stack_local(
        vv_path, vh_path, slope_path, elevation_path
    )

    # Train
    clf, valid_mask, report = train_classifier_local(stack, labels_2d)

    # Classify full image
    print("\nClassifying image...")
    class_map = classify_image_local(clf, stack)

    # Save everything
    save_results_local(class_map, profile, output_dir, date_str, vv_path)

    print(f"\n{'='*60}")
    print("Done!  Results saved to:", output_dir)
    print(f"{'='*60}\n")
    return class_map, clf


# ═══════════════════════════════════════════════════════════════
# SYNTHETIC DEMO — spatially coherent SAR simulation
# ═══════════════════════════════════════════════════════════════

def _make_spatial_region(H, W, seed=0):
    """
    Generate a smooth 2-D spatial mask using perlin-like noise.
    Returns a float array in [0, 1] with blob-like structure.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    base = rng.random((H, W))
    # Multi-scale smoothing → natural blob shapes
    smooth = (gaussian_filter(base, sigma=20) * 0.6 +
              gaussian_filter(base, sigma=5)  * 0.4)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    return smooth.astype(np.float32)


def _synthetic_demo():
    """
    Simulate a realistic Sentinel-1 SAR scene and classify it.

    SAR backscatter physics:
      Non-Glacier (rough terrain)  → high VV (-8 dB), medium VH (-16 dB)
      Glacier (smooth ice/snow)    → medium VV (-14 dB), low VH (-22 dB)
      Glacier Lake (specular)      → very low VV (-20 dB), very low VH (-28 dB)
    """
    from scipy.ndimage import gaussian_filter

    print("Generating spatially coherent SAR scene …")
    H, W = 512, 512
    rng  = np.random.default_rng(42)

    # ── 1. Build spatial class map (truth) ──────────────────────
    # Glacier occupies a connected highland blob in the centre
    glacier_field  = _make_spatial_region(H, W, seed=1)
    lake_field     = _make_spatial_region(H, W, seed=7)

    truth = np.zeros((H, W), dtype=np.int32)          # 0 = Non-Glacier
    truth[glacier_field > 0.62] = 1                   # 1 = Glacier   (~25% area)
    # Lakes form at low edges of glaciers
    lake_zone = (lake_field > 0.74) & (truth == 0)
    truth[lake_zone] = 2                              # 2 = Glacier Lake (~8% area)

    # ── 2. Generate class-specific SAR backscatter ───────────────
    # Realistic dB means and stds from literature
    sar_params = {
        #        VV_mean  VV_std  VH_mean  VH_std  slope_mean  slope_std
        0: dict(vv=-8,   sv=3,   vh=-16,  sh=2.5, sl=25,      ss=8),   # terrain
        1: dict(vv=-14,  sv=2,   vh=-22,  sh=1.5, sl=12,      ss=5),   # glacier
        2: dict(vv=-20,  sv=1,   vh=-28,  sh=1,   sl=2,       ss=1),   # lake
    }

    vv_img    = np.zeros((H, W), dtype=np.float32)
    vh_img    = np.zeros((H, W), dtype=np.float32)
    slope_img = np.zeros((H, W), dtype=np.float32)

    for cls, p in sar_params.items():
        mask = truth == cls
        vv_img[mask]    = rng.normal(p['vv'], p['sv'], mask.sum())
        vh_img[mask]    = rng.normal(p['vh'], p['sh'], mask.sum())
        slope_img[mask] = rng.normal(p['sl'], p['ss'], mask.sum()).clip(0, 90)

    # Add spatially correlated speckle noise (SAR characteristic)
    speckle = gaussian_filter(rng.normal(0, 1.5, (H, W)).astype(np.float32), sigma=1)
    vv_img  += speckle
    vh_img  += speckle * 0.7

    # ── 3. Build feature stack ───────────────────────────────────
    ratio = np.where(vv_img != 0, vh_img / vv_img, 0).astype(np.float32)
    stack = np.dstack([vv_img, vh_img, ratio, slope_img])   # (H, W, 4)

    # ── 4. Training labels (use 40% of pixels, leave rest unlabelled) ──
    sample_mask = rng.random((H, W)) < 0.40
    labels_2d   = np.full((H, W), 255, dtype=np.int32)      # 255 = unlabelled
    labels_2d[sample_mask] = truth[sample_mask]

    # ── 5. Train & classify ──────────────────────────────────────
    print("Training 3-class Random Forest on 40% sampled labels …")
    clf, _, report = train_classifier_local(stack, labels_2d)

    print("\nClassifying full image …")
    class_map = classify_image_local(clf, stack)

    # ── 6. Professional map-style visualisation ──────────────────
    print("\nRendering map output …")

    cmap = ListedColormap([CLASS_COLORS[i] for i in sorted(CLASS_COLORS)])
    legend_patches = [
        Patch(facecolor=CLASS_COLORS[i], edgecolor='#333',
              linewidth=0.5, label=CLASS_LABELS[i])
        for i in sorted(CLASS_LABELS)
    ]

    fig = plt.figure(figsize=(18, 6.5), facecolor='#1a1a2e')
    fig.suptitle(
        'Sentinel-1 SAR  ·  3-Class Glacier Mapping',
        color='white', fontsize=15, fontweight='bold', y=1.01
    )

    # Panel titles style
    title_kw = dict(fontsize=11, fontweight='bold', color='white', pad=8)

    # ── Panel 1: VV Backscatter (input) ─────────────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    vv_clip = np.clip(vv_img, -28, -2)
    im1 = ax1.imshow(vv_clip, cmap='gray', interpolation='bilinear')
    ax1.set_title('Sentinel-1 VV Backscatter (dB)', **title_kw)
    ax1.set_xlabel('Easting  →', color='#aaa', fontsize=8)
    ax1.set_ylabel('Northing  ↑', color='#aaa', fontsize=8)
    ax1.tick_params(colors='#888', labelsize=7)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#555')
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal',
                         pad=0.04, fraction=0.046)
    cbar1.ax.tick_params(colors='#aaa', labelsize=7)
    cbar1.set_label('dB', color='#aaa', fontsize=8)

    # ── Panel 2: Classified Map (output) ────────────────────────
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(class_map, cmap=cmap, vmin=0, vmax=2,
                     interpolation='nearest')
    ax2.set_title('3-Class Classification Map', **title_kw)
    ax2.set_xlabel('Easting  →', color='#aaa', fontsize=8)
    ax2.set_ylabel('Northing  ↑', color='#aaa', fontsize=8)
    ax2.tick_params(colors='#888', labelsize=7)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#555')
    leg = ax2.legend(handles=legend_patches, loc='lower right',
                     facecolor='#111', edgecolor='#555',
                     labelcolor='white', fontsize=8.5,
                     title='Class', title_fontsize=8)
    leg.get_title().set_color('white')

    # ── Panel 3: Overlay (SAR + classification alpha) ────────────
    ax3 = fig.add_subplot(1, 3, 3)
    # Normalise VV for display
    vv_norm = (vv_clip - vv_clip.min()) / (vv_clip.max() - vv_clip.min())
    # Build RGBA overlay
    rgba_map = np.zeros((H, W, 4), dtype=np.float32)
    rgba_map[class_map == 0] = [0.55, 0.42, 0.08, 0.50]   # brown, semi-transparent
    rgba_map[class_map == 1] = [0.91, 0.96, 0.99, 0.65]   # icy white
    rgba_map[class_map == 2] = [0.10, 0.50, 0.83, 0.75]   # blue
    ax3.imshow(vv_norm, cmap='gray', interpolation='bilinear')
    ax3.imshow(rgba_map, interpolation='nearest')
    ax3.set_title('Classification Overlay on SAR', **title_kw)
    ax3.set_xlabel('Easting  →', color='#aaa', fontsize=8)
    ax3.set_ylabel('Northing  ↑', color='#aaa', fontsize=8)
    ax3.tick_params(colors='#888', labelsize=7)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#555')
    leg3 = ax3.legend(handles=legend_patches, loc='lower right',
                      facecolor='#111', edgecolor='#555',
                      labelcolor='white', fontsize=8.5)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#0d0d1a')

    fig.patch.set_facecolor('#1a1a2e')
    plt.tight_layout()
    out_png = 'S1_classification_map.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"\n✓ Map saved to: {out_png}")
    print("  Left  → VV Backscatter (SAR input)")
    print("  Centre → 3-Class classified map")
    print("  Right  → Classification overlay on SAR")


# ═══════════════════════════════════════════════════════════════
# REAL DATA RUNNER — uses Sentinel1_1 folder GeoTIFFs
# ═══════════════════════════════════════════════════════════════

def run_on_real_s1(tif_path, output_dir='classified_glacier_maps',
                   sample_frac=0.30):
    """
    Load a real Sentinel-1 GeoTIFF (bands: VV, VH, angle),
    auto-generate training labels via SAR backscatter thresholds,
    train a 3-class Random Forest, and produce a map-style output.

    Thresholds based on C-band IW GRD backscatter physics:
      Glacier Lake  (specular)  → VV < -16 dB  AND  VH < -25 dB
      Glacier       (snow/ice)  → VV in [-16, -10]  AND  VH < -20 dB
      Non-Glacier   (terrain)   → everything else
    """
    from scipy.ndimage import gaussian_filter

    date_str = os.path.basename(tif_path)[3:13]   # e.g. '2023-07-18'
    print(f"\n{'='*62}")
    print(f"Sentinel-1  ·  3-Class Classifier  ·  {date_str}")
    print(f"{'='*62}")
    print(f"File: {tif_path}")

    # ── 1. Load bands ──────────────────────────────────────
    with rasterio.open(tif_path) as src:
        vv      = src.read(1).astype(np.float32)   # dB
        vh      = src.read(2).astype(np.float32)   # dB
        angle   = src.read(3).astype(np.float32)   # incidence angle
        profile = src.profile
        nodata  = src.nodata

    H, W = vv.shape
    print(f"Image size: {W} x {H} pixels")

    # Mask nodata
    valid_px = np.ones((H, W), dtype=bool)
    if nodata is not None:
        valid_px &= (vv != nodata) & (vh != nodata)
    valid_px &= np.isfinite(vv) & np.isfinite(vh)

    # ── 2. Feature engineering ─────────────────────────────
    ratio = np.where(vv != 0, vh / vv, 0).astype(np.float32)   # VH/VV
    # Smoothed bands reduce speckle (used for labelling thresholds)
    vv_sm = gaussian_filter(vv, sigma=2.5)
    vh_sm = gaussian_filter(vh, sigma=2.5)
    # Local texture (approx std dev via variance trick, 5×5 equiv)
    vv_sq   = gaussian_filter(vv**2, sigma=2.5)
    texture = np.sqrt(np.maximum(vv_sq - vv_sm**2, 0)).astype(np.float32)

    stack = np.dstack([vv, vh, ratio, angle,
                       vv_sm, vh_sm, texture])     # (H, W, 7)

    # ── 3. Threshold-based auto-labelling ─────────────────────
    # Physics-based thresholds for C-band IW GRD (dB):
    #   Glacier Lake: very low VV AND VH (specular open water)
    #   Glacier:      medium-low VV, low VH (snow/firn)
    #   Non-Glacier:  everything else (rough terrain, shadowed slopes)
    from scipy.ndimage import label as ndlabel
    from skimage.measure import regionprops

    # ── Tunable parameters ─────────────────────────────────────
    # Tighter thresholds seed only the most confidently water-like pixels.
    # The RF then extends the boundary into ambiguous pixels cleanly.
    vv_lake_thr  = -15.0   # VV threshold for lake seed pixels
    vh_lake_thr  = -24.0   # VH threshold for lake seed pixels
    vv_glac_lo   = -15.0   # VV lower bound for glacier
    vv_glac_hi   =  -9.0   # VV upper bound for glacier
    vh_glac_thr  = -19.0   # VH threshold for glacier

    min_lake_px  = 2000    # minimum blob area [pixels]
    # Shape: shadows are elongated (high eccentricity) AND irregular (low solidity)
    # A blob is a SHADOW if BOTH conditions hold simultaneously
    shadow_ecc_min  = 0.96   # eccentricity above this is suspicious
    shadow_sol_max  = 0.55   # solidity below this is suspicious

    lake_raw     = valid_px & (vv_sm < vv_lake_thr) & (vh_sm < vh_lake_thr)
    glacier_mask = (valid_px & (vv_sm >= vv_glac_lo) & (vv_sm < vv_glac_hi)
                   & (vh_sm < vh_glac_thr) & ~lake_raw)
    terrain_mask = valid_px & ~lake_raw & ~glacier_mask

    # ── Stage 1: remove tiny blobs (speckle / noise) ──────────
    labeled_lakes, n_blobs = ndlabel(lake_raw)
    blob_sizes = np.bincount(labeled_lakes.ravel())
    blob_sizes[0] = 0
    size_keep = set(int(i) for i in np.where(blob_sizes >= min_lake_px)[0])

    # ── Stage 2: shape filter — reject elongated shadow strips ─
    # Radar shadows cast along ridgelines are BOTH highly elongated
    # (eccentricity → 1) AND geometrically irregular (low solidity).
    # A real lake can be somewhat elongated OR somewhat irregular, but
    # NOT both at once.  We reject a blob only when BOTH criteria exceed
    # the thresholds simultaneously.
    shape_keep     = set()
    n_shadow_shape = 0
    for p in regionprops(labeled_lakes):
        if p.label not in size_keep:
            continue
        is_shadow = (p.eccentricity >= shadow_ecc_min and
                     p.solidity     <= shadow_sol_max)
        if is_shadow:
            n_shadow_shape += 1
        else:
            shape_keep.add(p.label)

    lake_mask    = np.isin(labeled_lakes, list(shape_keep))
    shadow_mask  = lake_raw & ~lake_mask
    terrain_mask = terrain_mask | shadow_mask

    n_lake    = int(lake_mask.sum())
    n_glacier = int(glacier_mask.sum())
    n_terrain = int(terrain_mask.sum())
    n_shadow  = int(shadow_mask.sum())
    n_total   = n_lake + n_glacier + n_terrain
    n_kept    = len(shape_keep)
    print(f"\nAuto-label summary ({n_total:,} labelled px):")
    print(f"  Non-Glacier : {n_terrain:>10,}  ({n_terrain/n_total*100:.1f}%)")
    print(f"  Glacier     : {n_glacier:>10,}  ({n_glacier/n_total*100:.1f}%)")
    print(f"  Glacier Lake: {n_lake:>10,}  ({n_lake/n_total*100:.1f}%)")
    print(f"  Shadow filter: {n_blobs} raw → {len(size_keep)} passed size "
          f"→ {n_kept} passed shape ({n_shadow_shape} elongated shadows removed)")

    if n_glacier < 100 or n_lake < 50:
        print("\n⚠ Very few glacier/lake pixels found with default thresholds.")
        print("  Adjust thresholds in run_on_real_s1() if needed.")

    # Sub-sample to speed up training
    rng = np.random.default_rng(42)
    labels_train = np.full((H, W), 255, dtype=np.int32)
    for cls, msk in [(0, terrain_mask), (1, glacier_mask), (2, lake_mask)]:
        coords = np.argwhere(msk)
        if len(coords) == 0:
            continue
        n_sample = max(500, int(len(coords) * sample_frac))
        n_sample = min(n_sample, len(coords))
        chosen = rng.choice(len(coords), n_sample, replace=False)
        for y, x in coords[chosen]:
            labels_train[y, x] = cls

    # ── 4. Train + classify ─────────────────────────────────
    print(f"\nTraining Random Forest on {(labels_train != 255).sum():,} sampled pixels…")
    clf, _, report = train_classifier_local(stack, labels_train)

    print("\nClassifying full image…")
    class_map = classify_image_local(clf, stack)
    # Mask nodata back to 0
    class_map[~valid_px] = 0

    # ── 5. Save GeoTiff + GeoJSONs ───────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    tif_out  = os.path.join(output_dir, f'S1_classified_{date_str}.tif')
    tif_prof = profile.copy()
    tif_prof.update({'dtype': rasterio.uint8, 'count': 1, 'compress': 'lzw', 'nodata': None})
    with rasterio.open(tif_out, 'w', **tif_prof) as dst:
        dst.write(class_map.astype(np.uint8), 1)
        dst.update_tags(CLASS_0='Non-Glacier', CLASS_1='Glacier', CLASS_2='Glacier Lake')
    print(f"\n✓ GeoTiff  : {tif_out}")

    from rasterio.features import shapes
    for cls_val, cls_name in CLASS_LABELS.items():
        binary = (class_map == cls_val).astype(np.uint8)
        if not binary.any():
            continue
        polys = [
            {'geometry': g, 'properties': {'class_id': cls_val, 'class_name': cls_name, 'date': date_str}}
            for g, v in shapes(binary, mask=binary.astype(bool), transform=profile['transform'])
            if v == 1
        ]
        if polys:
            import geopandas as gpd
            gdf = gpd.GeoDataFrame.from_features(polys, crs=profile['crs'])
            vec_out = os.path.join(output_dir, f'S1_{cls_name.replace(" ","_")}_{date_str}.geojson')
            gdf.to_file(vec_out, driver='GeoJSON')
            print(f"✓ GeoJSON  : {vec_out}  ({len(gdf)} polygons)")

    # ── 6. Map-style visualisation ────────────────────────────
    cmap = ListedColormap([CLASS_COLORS[i] for i in sorted(CLASS_COLORS)])
    legend_patches = [
        Patch(facecolor=CLASS_COLORS[i], edgecolor='#333', linewidth=0.5,
              label=CLASS_LABELS[i]) for i in sorted(CLASS_LABELS)
    ]
    title_kw = dict(fontsize=11, fontweight='bold', color='white', pad=8)

    # Clip VV for display
    p2, p98 = np.nanpercentile(vv[valid_px], [2, 98])
    vv_disp = np.clip(vv, p2, p98)
    vv_norm = (vv_disp - vv_disp.min()) / (vv_disp.max() - vv_disp.min() + 1e-9)

    fig = plt.figure(figsize=(19, 7), facecolor='#1a1a2e')
    fig.suptitle(
        f'Sentinel-1  ·  3-Class Glacier Mapping  ·  {date_str}',
        color='white', fontsize=14, fontweight='bold', y=1.005
    )

    # Panel 1 — VV Backscatter
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(vv_disp, cmap='gray', interpolation='bilinear')
    ax1.set_title('Sentinel-1 VV Backscatter (dB)', **title_kw)
    ax1.set_xlabel('Easting  →', color='#aaa', fontsize=8)
    ax1.set_ylabel('Northing  ↑', color='#aaa', fontsize=8)
    ax1.tick_params(colors='#888', labelsize=7)
    for sp in ax1.spines.values(): sp.set_edgecolor('#555')
    cb1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.04, fraction=0.046)
    cb1.ax.tick_params(colors='#aaa', labelsize=7)
    cb1.set_label('dB', color='#aaa', fontsize=8)

    # Panel 2 — Classified Map
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(class_map, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
    ax2.set_title('3-Class Classification Map', **title_kw)
    ax2.set_xlabel('Easting  →', color='#aaa', fontsize=8)
    ax2.set_ylabel('Northing  ↑', color='#aaa', fontsize=8)
    ax2.tick_params(colors='#888', labelsize=7)
    for sp in ax2.spines.values(): sp.set_edgecolor('#555')
    leg = ax2.legend(handles=legend_patches, loc='lower right',
                     facecolor='#111', edgecolor='#555',
                     labelcolor='white', fontsize=8.5,
                     title='Class', title_fontsize=8)
    leg.get_title().set_color('white')

    # Panel 3 — Overlay
    ax3 = fig.add_subplot(1, 3, 3)
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[class_map == 0] = [0.55, 0.42, 0.08, 0.45]
    rgba[class_map == 1] = [0.91, 0.96, 0.99, 0.60]
    rgba[class_map == 2] = [0.10, 0.50, 0.83, 0.80]
    ax3.imshow(vv_norm, cmap='gray', interpolation='bilinear')
    ax3.imshow(rgba, interpolation='nearest')
    ax3.set_title('Classification Overlay on SAR', **title_kw)
    ax3.set_xlabel('Easting  →', color='#aaa', fontsize=8)
    ax3.set_ylabel('Northing  ↑', color='#aaa', fontsize=8)
    ax3.tick_params(colors='#888', labelsize=7)
    for sp in ax3.spines.values(): sp.set_edgecolor('#555')
    leg3 = ax3.legend(handles=legend_patches, loc='lower right',
                      facecolor='#111', edgecolor='#555',
                      labelcolor='white', fontsize=8.5)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#0d0d1a')

    fig.patch.set_facecolor('#1a1a2e')
    plt.tight_layout()
    png_out = os.path.join(output_dir, f'S1_classification_map_{date_str}.png')
    plt.savefig(png_out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    print(f"✓ Map PNG  : {png_out}")
    return class_map, clf


if __name__ == '__main__':
    S1_DIR    = r'f:\NIDM-SRM Drone Project Docs\Sentinel1_1'
    OUT_DIR   = r'f:\NIDM-SRM Drone Project Docs\classified_glacier_maps'

    # Pick the peak-melt image (July) for best glacier/lake contrast
    tif_path  = os.path.join(S1_DIR, 'S1_2023-07-18.tif')

    print("sentinel1_processing.py  —  3-Class Glacier Classifier")
    print("Classes: 0=Non-Glacier | 1=Glacier | 2=Glacier Lake\n")
    run_on_real_s1(tif_path, output_dir=OUT_DIR)
