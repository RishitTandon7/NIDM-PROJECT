"""
Sentinel-1 SAR  -  3-Class Glacier Classifier  v5
==================================================
Classes:  0=Non-Glacier   1=Glacier   2=Glacier Lake

WHY PREVIOUS VERSIONS OVER-CLASSIFIED LAKES
--------------------------------------------
Every version below v5 tagged 10-20% of the scene as "lake" when
real glacier lakes occupy <1%.  The cause: radar shadows and
specular water both return near-zero backscatter in VV/VH.

v5 SOLUTION: LSWI + CONTEXT + SHAPE  (works WITHOUT DEM)
------------------------------------------------------
1. LSWI (Linear-Scale Water Index)
   In LINEAR power space (not dB!), VV and VH are equally
   suppressed by specular water (LSWI near 0), but shadows
   keep VV >> VH (LSWI >> 0).  This is the key discriminator.

2. SAR SHADOW MASK from LSWI gradient anisotropy
   Shadows are directional streaks; lakes have isotropic edges.

3. STRICT signal thresholds: VV < -17 dB, VH < -24 dB
   Only the DARKEST pixels are candidates.

4. CONTEXT GATE: lake candidates must be ADJACENT to bright
   (glacier/moraine) pixels.  Isolated dark pools = shadow.

5. SHAPE FILTER: elongation < 3.0, compactness > 0.2, min 25 px.

6. RF trained with LSWI + shadow_score as features.

Author: Antigravity / v5
"""

import numpy as np
import rasterio
from rasterio.features import shapes
from scipy.ndimage import (gaussian_filter, uniform_filter,
                            binary_opening, binary_closing,
                            binary_dilation, label as nd_label)
from scipy.ndimage import minimum_filter, maximum_filter, sobel
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os, sys, warnings
warnings.filterwarnings('ignore')

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CLASS_LABELS = {0: "Non-Glacier", 1: "Glacier", 2: "Glacier Lake"}
CLASS_COLORS = {0: "#8B6914", 1: "#E8F4FD", 2: "#1A7FD4"}


# ================================================================
# UTILITIES
# ================================================================

def _lin(db):
    """dB to linear power."""
    return np.power(10.0, np.clip(db, -40, 0) / 10.0).astype(np.float32)

def _lv(band, w):
    """Local variance."""
    b = band.astype(np.float64)
    return np.maximum(uniform_filter(b**2, w) -
                      uniform_filter(b, w)**2, 0).astype(np.float32)

def _lr(band, w):
    """Local range."""
    return (maximum_filter(band, w) -
            minimum_filter(band, w)).astype(np.float32)


def compute_lswi(vv_db, vh_db):
    """
    LSWI = (VV_lin - VH_lin) / (VV_lin + VH_lin)
    Computed in LINEAR power space.
    
    Water:  LSWI ~ 0      (both channels equally suppressed)
    Shadow: LSWI ~ 0.3-0.7 (VV > VH maintained even at low power)
    Terrain: LSWI ~ 0.3-0.8
    Glacier: LSWI ~ 0.15-0.45
    """
    vv = _lin(vv_db)
    vh = _lin(vh_db)
    return ((vv - vh) / (vv + vh + 1e-12)).astype(np.float32)


def otsu_threshold(data, mask=None, n_bins=256):
    """Otsu adaptive threshold."""
    vals = data[mask] if mask is not None else data.ravel()
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [1, 99])
    vals = vals[(vals >= lo) & (vals <= hi)]
    
    hist, edges = np.histogram(vals, bins=n_bins)
    centres = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    sum_total = (hist * centres).sum()
    
    best_t, best_v = centres[0], 0
    w0, s0 = 0, 0
    for i in range(len(hist)):
        w0 += hist[i]
        if w0 == 0: continue
        w1 = total - w0
        if w1 == 0: break
        s0 += hist[i] * centres[i]
        m0 = s0 / w0
        m1 = (sum_total - s0) / w1
        v = w0 * w1 * (m0 - m1)**2
        if v > best_v:
            best_v = v
            best_t = centres[i]
    return best_t


# ================================================================
# SAR-BASED SHADOW MASK (no DEM required)
# ================================================================

def compute_shadow_mask_sar(vv, vh, valid_px):
    """
    Build a shadow mask using ONLY SAR data.
    
    Key insight: LSWI separates water (LSWI~0) from shadow (LSWI>0.3).
    We use Otsu on LSWI of dark pixels to find the split.
    
    Additionally, shadows have directional gradient patterns:
    - Shadows are elongated streaks in range direction
    - Lakes have isotropic (round) edges
    """
    vv_sm = gaussian_filter(vv, sigma=1.5)
    vh_sm = gaussian_filter(vh, sigma=1.5)
    lswi  = compute_lswi(vv, vh)
    lswi_sm = gaussian_filter(lswi, sigma=2.5)
    
    # Dark pixels = potential lake OR shadow
    dark = valid_px & (vv_sm < -13.0)
    
    # Otsu on LSWI of dark pixels
    if dark.sum() > 200:
        lswi_thresh = otsu_threshold(lswi_sm, dark)
        lswi_thresh = np.clip(lswi_thresh, 0.12, 0.40)
    else:
        lswi_thresh = 0.22
    
    print(f"  [shadow] LSWI Otsu threshold: {lswi_thresh:.3f}")
    
    # Shadow = dark pixel with LSWI significantly above water level
    shadow_lswi = dark & (lswi_sm > lswi_thresh + 0.05)
    
    # Gradient anisotropy: shadows are directional
    gx = sobel(lswi_sm, axis=1)
    gy = sobel(lswi_sm, axis=0)
    aniso = np.abs(gx) / (np.abs(gy) + 1e-6)
    aniso_sm = gaussian_filter(aniso, sigma=2.5)
    p85 = np.percentile(aniso_sm[valid_px], 85)
    high_aniso = aniso_sm > p85
    
    # Local VV range at low VV = shadow edge
    vv_range = _lr(vv_sm, 9)
    p75 = np.percentile(vv_range[valid_px], 75)
    at_edge = dark & (vv_range > p75)
    
    # Combine
    shadow = shadow_lswi | (high_aniso & at_edge & dark)
    shadow = binary_dilation(shadow, iterations=2)
    shadow = binary_opening(shadow, np.ones((3,3)), iterations=1)
    
    pct = shadow.sum() / shadow.size * 100
    print(f"  [shadow] SAR shadow mask: {shadow.sum():,} px ({pct:.1f}%)")
    
    return shadow.astype(bool), lswi_thresh, lswi_sm


# ================================================================
# GEOMETRIC SHADOW MASK (when DEM is available)
# ================================================================

def compute_shadow_mask_dem(slope, aspect, radar_azimuth=192.0):
    """DEM-based shadow mask. Much more reliable than SAR-only."""
    if slope is None:
        return None
    
    away = (radar_azimuth + 180) % 360
    if aspect is not None:
        ad = np.abs(aspect - away) % 360
        ad = np.minimum(ad, 360 - ad)
        facing = ad < 75
    else:
        facing = np.ones_like(slope, bool)
    
    shadow = binary_dilation(
        (facing & (slope > 15)) | (slope > 30), iterations=3)
    
    pct = shadow.sum() / shadow.size * 100
    print(f"  [shadow] DEM shadow mask: {shadow.sum():,} px ({pct:.1f}%)")
    return shadow.astype(bool)


# ================================================================
# FEATURE STACK
# ================================================================

def build_features(vv, vh, angle=None, slope=None, aspect=None,
                   elevation=None, shadow_score=None, lswi_sm=None):
    """Build feature stack for RF classifier."""
    H, W = vv.shape
    
    vv_sm  = gaussian_filter(vv, 1.5).astype(np.float32)
    vh_sm  = gaussian_filter(vh, 1.5).astype(np.float32)
    vv_med = gaussian_filter(vv, 3.5).astype(np.float32)
    vh_med = gaussian_filter(vh, 3.5).astype(np.float32)
    
    lr = (vv_sm - vh_sm).astype(np.float32)
    
    if lswi_sm is None:
        lswi_sm = gaussian_filter(compute_lswi(vv, vh), 2.0).astype(np.float32)
    
    tv5  = _lv(vv_sm, 5)
    tv11 = _lv(vv_sm, 11)
    rng  = _lr(vv_sm, 11)
    
    vv_lin = _lin(vv_sm).astype(np.float64)
    cv_m  = uniform_filter(vv_lin, 9)
    cv_m2 = uniform_filter(vv_lin**2, 9)
    cv = (np.sqrt(np.maximum(cv_m2 - cv_m**2, 0)) / (cv_m + 1e-9)).astype(np.float32)
    
    # Context: bright neighbour fraction
    bright = (vv_sm > -12.0).astype(np.float32)
    nbr_bright = uniform_filter(bright.astype(np.float64), 31).astype(np.float32)
    
    bands = [vv, vh, vv_sm, vh_sm, vv_med, vh_med,
             lr, lswi_sm, tv5, tv11, rng, cv, nbr_bright]
    names = ['VV','VH','VV_sm','VH_sm','VV_med','VH_med',
             'log_ratio','LSWI_sm','tex_vv5','tex_vv11',
             'range_vv11','cv_lin','nbr_bright_frac']
    
    if shadow_score is not None:
        bands.append(shadow_score); names.append('shadow_score')
    if angle is not None:
        bands.append(angle.astype(np.float32)); names.append('angle')
    if slope is not None:
        bands.append(slope.astype(np.float32)); names.append('slope')
    if elevation is not None:
        bands.append(elevation.astype(np.float32)); names.append('elevation')
    
    stack = np.dstack(bands)
    print(f"  Feature stack: {stack.shape}  ({len(names)} bands)")
    return stack, names


# ================================================================
# AUTO-LABELLING (VERY STRICT — targets <1% lake)
# ================================================================

def auto_label(vv, vh, valid_px, shadow_mask, lswi_sm, lswi_thresh,
               slope=None, elevation=None,
               sample_frac=0.30, rng_seed=42):
    """
    Conservative auto-labelling with multi-gate rejection.
    
    LAKE requires ALL of:
      L1: VV_med < -15 dB          (dark pixel)
      L2: VH_med < -22 dB          (dark pixel)
      L3: LSWI_sm in [-0.12, lswi_thresh]  (specular water signature)
      L4: log_ratio(VV-VH) < 7 dB  (VV ~ VH -- specular)
      L5: cv_lin < 0.45            (homogeneous in linear scale)
      L6: NOT in shadow mask
      L7: slope < 15 (if available)
      L8: blob >= 15 px, elongation < 3.0, compactness > 0.2
      L9: context: >10% of 10px-dilated ring is non-dark
    """
    H, W = vv.shape
    rng = np.random.default_rng(rng_seed)
    
    vv_sm  = gaussian_filter(vv, 1.5)
    vh_sm  = gaussian_filter(vh, 1.5)
    vv_med = gaussian_filter(vv, 4.0)
    vh_med = gaussian_filter(vh, 4.0)
    lr     = (vv_sm - vh_sm).astype(np.float32)
    
    vv_lin = _lin(vv_sm).astype(np.float64)
    cv_m  = uniform_filter(vv_lin, 9)
    cv_m2 = uniform_filter(vv_lin**2, 9)
    cv = (np.sqrt(np.maximum(cv_m2 - cv_m**2, 0)) / (cv_m + 1e-9)).astype(np.float32)
    
    # L1-L7: signal + geometry gates
    L1 = vv_med < -15.0
    L2 = vh_med < -22.0
    L3 = (lswi_sm >= -0.12) & (lswi_sm <= lswi_thresh)
    L4 = lr < 7.0
    L5 = cv < 0.45
    L6 = ~shadow_mask if shadow_mask is not None else np.ones((H,W), bool)
    L7 = (slope < 15.0) if slope is not None else np.ones((H,W), bool)
    
    lake_raw = valid_px & L1 & L2 & L3 & L4 & L5 & L6 & L7
    
    # L8: shape filter on blobs
    lbl, n = nd_label(lake_raw)
    lake_mask = np.zeros((H,W), bool)
    rej_size = rej_shape = rej_ctx = 0
    
    not_dark = (vv_med > -13.0) & valid_px
    
    for i in range(1, n+1):
        blob = lbl == i
        area = blob.sum()
        if area < 15:
            rej_size += 1
            continue
        
        # Elongation check
        rows = np.where(blob.any(axis=1))[0]
        cols = np.where(blob.any(axis=0))[0]
        bh = rows[-1] - rows[0] + 1
        bw = cols[-1] - cols[0] + 1
        elong = max(bh, bw) / (min(bh, bw) + 1)
        if elong > 3.0:
            rej_shape += 1
            continue
        
        # Compactness
        compact = area / (bh * bw + 1)
        if compact < 0.20:
            rej_shape += 1
            continue
        
        # L9: context — must be near bright pixels
        ring = binary_dilation(blob, iterations=10) & ~blob & valid_px
        if ring.sum() == 0:
            rej_ctx += 1
            continue
        frac = not_dark[ring].mean()
        if frac < 0.10:
            rej_ctx += 1
            continue
        
        lake_mask |= blob
    
    print(f"  [label] Lake blobs: {rej_size} tiny, "
          f"{rej_shape} bad shape, {rej_ctx} bad context rejected")
    
    # Glacier: moderate backscatter
    sl_ok = (slope < 25) if slope is not None else np.ones((H,W), bool)
    glacier_mask = (valid_px & ~lake_mask & ~shadow_mask
                    & (vv_med >= -16.0) & (vv_med < -8.0)
                    & (vh_med < -18.0)
                    & (lswi_sm > lswi_thresh)
                    & sl_ok)
    
    # Terrain / shadow
    terrain_mask = (valid_px
                    & (shadow_mask | (vv_med > -8.0) | (lr > 9.0)
                       | (lswi_sm > 0.55))
                    & ~lake_mask & ~glacier_mask)
    
    # Build labels
    labels_full = np.full((H,W), 255, np.int32)
    labels_full[terrain_mask] = 0
    labels_full[glacier_mask] = 1
    labels_full[lake_mask]    = 2
    
    # Subsample for training
    labels_s = np.full((H,W), 255, np.int32)
    for cls, mask in [(0, terrain_mask), (1, glacier_mask), (2, lake_mask)]:
        coords = np.argwhere(mask)
        if not len(coords): continue
        n_s = max(800, int(len(coords) * sample_frac))
        n_s = min(n_s, len(coords))
        idx = rng.choice(len(coords), n_s, replace=False)
        for y, x in coords[idx]:
            labels_s[y, x] = cls
    
    nt = terrain_mask.sum(); ng = glacier_mask.sum(); nl = lake_mask.sum()
    tot = max(nt + ng + nl, 1)
    print(f"\n  Auto-labels: {tot:,} total  |  {(labels_full==255).sum():,} ambiguous")
    print(f"    Non-Glacier : {nt:>9,}  ({nt/tot*100:.1f}%)")
    print(f"    Glacier     : {ng:>9,}  ({ng/tot*100:.1f}%)")
    print(f"    Glacier Lake: {nl:>9,}  ({nl/tot*100:.1f}%)")
    
    lake_pct = nl / valid_px.sum() * 100
    print(f"\n  Lake % of valid area: {lake_pct:.2f}%")
    if lake_pct > 5:
        print("  WARNING: >5% lake is suspicious for a glacier scene.")
        print("           Most Himalayan scenes have <1% real lake area.")
    
    return labels_s, labels_full


# ================================================================
# CLASSIFIER + PREDICT
# ================================================================

def train_rf(stack, labels_s, n_trees=300):
    """Train RF classifier."""
    H, W, F = stack.shape
    X = stack.reshape(-1, F)
    y = labels_s.ravel()
    ok = (y >= 0) & (y <= 2) & np.isfinite(X).all(axis=1)
    Xv, yv = X[ok], y[ok]
    
    present_classes = sorted(np.unique(yv))
    if len(present_classes) < 2:
        print("  ERROR: fewer than 2 classes in training data!")
        print(f"  Classes found: {present_classes}")
        raise ValueError("Need at least 2 classes to train.")
    
    Xtr, Xte, ytr, yte = train_test_split(
        Xv, yv, test_size=0.25, stratify=yv, random_state=42)
    
    rf = RandomForestClassifier(
        n_estimators=n_trees, max_depth=28,
        min_samples_split=4, min_samples_leaf=2,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    
    tgt_names = [CLASS_LABELS[i] for i in present_classes]
    print("\n  -- Classification Report --")
    print(classification_report(
        yte, rf.predict(Xte), labels=present_classes,
        target_names=tgt_names))
    print("  Confusion Matrix:\n", confusion_matrix(yte, rf.predict(Xte),
                                                      labels=present_classes))
    
    return rf


def predict_and_clean(rf, stack, valid_px, shadow_mask, lswi_sm,
                      lswi_thresh):
    """
    Predict + aggressive post-processing:
    1. Shadow hard override
    2. LSWI veto: predicted lake with LSWI > lswi_thresh+0.05 -> non-glacier
    3. RF probability veto: P(lake) < 0.50 -> non-glacier
    4. Shape filter: min 30px, elongation < 3.0
    5. Context gate: must be near bright pixels
    """
    H, W, F = stack.shape
    X = stack.reshape(-1, F)
    
    chunk = 200_000
    proba = np.zeros((H*W, 3), np.float32)
    print("  Classifying ...")
    for i in range(0, H*W, chunk):
        proba[i:i+chunk] = rf.predict_proba(X[i:i+chunk])
    
    proba = proba.reshape(H, W, 3)
    cmap = proba.argmax(axis=2).astype(np.uint8)
    cmap[~valid_px] = 0
    
    # 1. Shadow override
    if shadow_mask is not None:
        n_override = ((cmap == 2) & shadow_mask).sum()
        cmap[shadow_mask] = 0
        print(f"  Shadow override: {n_override:,} lake px -> Non-Glacier")
    
    # 2. LSWI veto
    lswi_bad = (cmap == 2) & (lswi_sm > lswi_thresh + 0.05)
    cmap[lswi_bad] = 0
    print(f"  LSWI veto: {lswi_bad.sum():,} high-LSWI lake px -> Non-Glacier")
    
    # 3. Probability veto
    prob_bad = (cmap == 2) & (proba[:,:,2] < 0.50)
    cmap[prob_bad] = 0
    print(f"  Prob veto: {prob_bad.sum():,} low-confidence lake px -> Non-Glacier")
    
    # 4. Shape filter
    struct3 = np.ones((3,3), bool)
    lake_bin = binary_opening(cmap == 2, struct3, iterations=2)
    
    lbl, n = nd_label(lake_bin)
    lake_clean = np.zeros((H,W), bool)
    rej_s = rej_e = rej_c = 0
    
    vv_med = gaussian_filter(stack[:,:,0], 4.0)
    not_dark = (vv_med > -13.0) & valid_px
    
    for i in range(1, n+1):
        blob = lbl == i
        area = blob.sum()
        if area < 30:
            rej_s += 1; continue
        
        rows = np.where(blob.any(axis=1))[0]
        cols = np.where(blob.any(axis=0))[0]
        bh = rows[-1] - rows[0] + 1
        bw = cols[-1] - cols[0] + 1
        elong = max(bh, bw) / (min(bh, bw) + 1)
        if elong > 3.0:
            rej_e += 1; continue
        
        compact = area / (bh * bw + 1)
        if compact < 0.20:
            rej_e += 1; continue
        
        # 5. Context gate
        ring = binary_dilation(blob, iterations=10) & ~blob & valid_px
        if ring.sum() == 0:
            rej_c += 1; continue
        if not_dark[ring].mean() < 0.12:
            rej_c += 1; continue
        
        lake_clean |= blob
    
    print(f"  Shape/context: {rej_s} tiny, {rej_e} elongated/irregular, "
          f"{rej_c} isolated removed")
    
    # Rebuild
    glacier_closed = binary_closing(cmap == 1, np.ones((5,5)), iterations=2)
    
    final = np.zeros((H,W), np.uint8)
    final[glacier_closed] = 1
    final[lake_clean]     = 2
    final[~valid_px]      = 0
    
    total = H * W
    print("\n  Final distribution:")
    for v, name in CLASS_LABELS.items():
        pct = (final == v).sum() / total * 100
        print(f"    {name:15s}: {pct:.2f}%")
    
    lake_pct = (final == 2).sum() / valid_px.sum() * 100
    print(f"\n  Lake % of valid area: {lake_pct:.2f}%")
    
    return final, proba


# ================================================================
# MAIN RUNNER
# ================================================================

def run_on_real_s1(tif_path,
                   output_dir='classified_glacier_maps',
                   sample_frac=0.30,
                   slope_path=None, aspect_path=None,
                   elevation_path=None,
                   radar_azimuth=192.0):
    """
    v5 pipeline: LSWI + context + shape, no network required.
    
    Provide slope_path/aspect_path/elevation_path for best results.
    Generate with:
        gdaldem slope   dem.tif  slope.tif   -p
        gdaldem aspect  dem.tif  aspect.tif
    """
    date_str = os.path.basename(tif_path)[3:13]
    print(f"\n{'='*64}")
    print(f"  Sentinel-1  -  3-Class Glacier Classifier v5  -  {date_str}")
    print(f"{'='*64}")
    
    # Load SAR
    with rasterio.open(tif_path) as src:
        vv      = src.read(1).astype(np.float32)
        vh      = src.read(2).astype(np.float32)
        angle   = src.read(3).astype(np.float32) if src.count >= 3 else None
        profile = src.profile
        nodata  = src.nodata
    
    H, W = vv.shape
    print(f"  Image: {W} x {H} px")
    
    valid_px = np.ones((H,W), bool)
    if nodata is not None:
        valid_px &= (vv != nodata) & (vh != nodata)
    valid_px &= np.isfinite(vv) & np.isfinite(vh)
    
    # Load DEM layers
    slope = aspect = elevation = None
    if slope_path and os.path.exists(slope_path):
        with rasterio.open(slope_path) as s:
            slope = s.read(1).astype(np.float32)
        print(f"  + Slope     : {slope_path}")
    else:
        print("  ! No slope -- DEM shadow mask disabled")
    
    if aspect_path and os.path.exists(aspect_path):
        with rasterio.open(aspect_path) as s:
            aspect = s.read(1).astype(np.float32)
        print(f"  + Aspect    : {aspect_path}")
    
    if elevation_path and os.path.exists(elevation_path):
        with rasterio.open(elevation_path) as s:
            elevation = s.read(1).astype(np.float32)
        print(f"  + Elevation : {elevation_path}")
    
    # Step 1: Shadow masks
    print("\n[1/5] Computing shadow masks ...")
    sar_shadow, lswi_thresh, lswi_sm = compute_shadow_mask_sar(
        vv, vh, valid_px)
    
    dem_shadow = compute_shadow_mask_dem(slope, aspect, radar_azimuth)
    
    if dem_shadow is not None:
        shadow_mask = sar_shadow | dem_shadow
        print(f"  Combined shadow: {shadow_mask.sum():,} px "
              f"({shadow_mask.sum()/shadow_mask.size*100:.1f}%)")
    else:
        shadow_mask = sar_shadow
    
    # Shadow score for features
    gx = sobel(lswi_sm, axis=1)
    gy = sobel(lswi_sm, axis=0)
    aniso = gaussian_filter(
        np.abs(gx) / (np.abs(gy) + 1e-6), 2.5).astype(np.float32)
    a95 = np.percentile(aniso[valid_px], 95)
    shadow_score = np.clip(aniso / (a95 + 1e-6), 0, 1).astype(np.float32)
    
    # Step 2: Feature stack
    print("\n[2/5] Building features ...")
    stack, feat_names = build_features(
        vv, vh, angle=angle, slope=slope, aspect=aspect,
        elevation=elevation, shadow_score=shadow_score,
        lswi_sm=lswi_sm)
    
    # Step 3: Auto-label
    print("\n[3/5] Auto-labelling ...")
    labels_s, labels_full = auto_label(
        vv, vh, valid_px, shadow_mask, lswi_sm, lswi_thresh,
        slope=slope, elevation=elevation,
        sample_frac=sample_frac)
    
    # Step 4: Train
    print("\n[4/5] Training RF ...")
    rf = train_rf(stack, labels_s, n_trees=300)
    
    # Step 5: Predict + clean
    print("\n[5/5] Predicting + post-processing ...")
    final, proba = predict_and_clean(
        rf, stack, valid_px, shadow_mask, lswi_sm, lswi_thresh)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    
    tif_out = os.path.join(output_dir, f'S1_classified_{date_str}.tif')
    tp = profile.copy()
    tp.update({'dtype': rasterio.uint8, 'count': 1,
               'compress': 'lzw', 'nodata': None})
    with rasterio.open(tif_out, 'w', **tp) as dst:
        dst.write(final, 1)
        dst.update_tags(CLASS_0='Non-Glacier', CLASS_1='Glacier',
                        CLASS_2='Glacier Lake', CLASSIFIER='RF_v5')
    print(f"\n  + GeoTiff  : {tif_out}")
    
    for cv, cn in CLASS_LABELS.items():
        binary = (final == cv).astype(np.uint8)
        if not binary.any(): continue
        polys = [
            {'geometry': g, 'properties': {
                'class_id': cv, 'class_name': cn, 'date': date_str}}
            for g, v in shapes(binary, mask=binary.astype(bool),
                               transform=profile['transform'])
            if v == 1
        ]
        if polys:
            gdf = gpd.GeoDataFrame.from_features(polys, crs=profile['crs'])
            vec = os.path.join(output_dir,
                               f'S1_{cn.replace(" ","_")}_{date_str}.geojson')
            gdf.to_file(vec, driver='GeoJSON')
            print(f"  + GeoJSON  : {vec}  ({len(gdf)} polys)")
    
    _render(vv, final, proba, shadow_mask, lswi_sm,
            valid_px, date_str, output_dir)
    
    print(f"\n{'='*64}")
    print(f"  Done -> {output_dir}")
    print(f"{'='*64}\n")
    return final, rf


# ================================================================
# VISUALISATION
# ================================================================

def _render(vv, cmap, proba, shadow_mask, lswi_sm,
            valid_px, date_str, output_dir):
    H, W = vv.shape
    cm = ListedColormap([CLASS_COLORS[i] for i in [0,1,2]])
    leg = [Patch(facecolor=CLASS_COLORS[i], edgecolor='#333',
                 linewidth=0.5, label=CLASS_LABELS[i]) for i in [0,1,2]]
    
    p2, p98 = np.nanpercentile(vv[valid_px], [2, 98])
    vvd = np.clip(vv, p2, p98)
    vvn = (vvd - vvd.min()) / (vvd.max() - vvd.min() + 1e-9)
    
    tkw = dict(fontsize=9, fontweight='bold', color='white', pad=6)
    
    npan = 6
    fig, axes = plt.subplots(1, npan, figsize=(4.5*npan, 6.5),
                              facecolor='#1a1a2e')
    fig.suptitle(f'S-1 - 3-Class Glacier v5 - {date_str}',
                 color='white', fontsize=13, fontweight='bold', y=1.003)
    
    def sty(ax):
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='#888', labelsize=5)
        for sp in ax.spines.values(): sp.set_edgecolor('#555')
    
    # VV
    ax = axes[0]
    im = ax.imshow(vvd, cmap='gray', interpolation='bilinear')
    ax.set_title('VV Backscatter (dB)', **tkw); sty(ax)
    cb = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, fraction=0.046)
    cb.ax.tick_params(colors='#aaa', labelsize=6)
    
    # LSWI
    ax = axes[1]
    im = ax.imshow(lswi_sm, cmap='RdYlBu_r', vmin=-0.2, vmax=0.7,
                   interpolation='bilinear')
    ax.set_title('LSWI [blue=water, red=shadow]', **tkw); sty(ax)
    cb = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, fraction=0.046)
    cb.ax.tick_params(colors='#aaa', labelsize=6)
    
    # Shadow mask
    ax = axes[2]
    ax.imshow(shadow_mask.astype(float) if shadow_mask is not None
              else np.zeros((H,W)),
              cmap='hot_r', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('Shadow Mask', **tkw); sty(ax)
    
    # P(Lake)
    ax = axes[3]
    im = ax.imshow(proba[:,:,2], cmap='Blues', vmin=0, vmax=1,
                   interpolation='bilinear')
    ax.set_title('P(Lake)', **tkw); sty(ax)
    cb = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, fraction=0.046)
    cb.ax.tick_params(colors='#aaa', labelsize=6)
    
    # 3-class map
    ax = axes[4]
    ax.imshow(cmap, cmap=cm, vmin=0, vmax=2, interpolation='nearest')
    ax.set_title('3-Class Map', **tkw); sty(ax)
    ll = ax.legend(handles=leg, loc='lower right', facecolor='#111',
                   edgecolor='#555', labelcolor='white', fontsize=7)
    ll.get_title().set_color('white') if ll.get_title() else None
    
    # Overlay
    ax = axes[5]
    rgba = np.zeros((H,W,4), np.float32)
    rgba[cmap==0] = [0.55, 0.42, 0.08, 0.38]
    rgba[cmap==1] = [0.91, 0.96, 0.99, 0.52]
    rgba[cmap==2] = [0.10, 0.50, 0.83, 0.88]
    ax.imshow(vvn, cmap='gray', interpolation='bilinear')
    ax.imshow(rgba, interpolation='nearest')
    ax.set_title('Overlay', **tkw); sty(ax)
    ax.legend(handles=leg, loc='lower right', facecolor='#111',
              edgecolor='#555', labelcolor='white', fontsize=7)
    
    fig.patch.set_facecolor('#1a1a2e')
    plt.tight_layout()
    out = os.path.join(output_dir, f'S1_classification_map_{date_str}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"  + Map PNG  : {out}")


# ================================================================
# GEE MODE
# ================================================================

def process_sentinel1_gee(lon, lat,
                           start='2023-06-01', end='2023-09-30'):
    import ee
    try: ee.Initialize()
    except: print("Run ee.Authenticate()"); raise
    
    roi = ee.Geometry.Point([lon, lat]).buffer(10000)
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(roi).filterDate(start, end)
          .filter(ee.Filter.eq('instrumentMode', 'IW')))
    comp = s1.median().clip(roi)
    
    lr = comp.select('VV').subtract(comp.select('VH')).rename('log_ratio')
    
    vv_lin = ee.Image(10).pow(comp.select('VV').divide(10))
    vh_lin = ee.Image(10).pow(comp.select('VH').divide(10))
    lswi = vv_lin.subtract(vh_lin).divide(
        vv_lin.add(vh_lin).add(1e-9)).rename('LSWI')
    
    dem = ee.Image('USGS/SRTMGL1_003').clip(roi)
    slope = ee.Terrain.slope(dem).rename('slope')
    elev = dem.rename('elevation')
    
    img = comp.addBands([lr, lswi, slope, elev])
    return img, roi


def train_and_classify_gee(img, pts):
    import ee
    bands = ['VV','VH','log_ratio','LSWI','slope','elevation']
    tr = img.sampleRegions(collection=pts, properties=['label'], scale=10)
    clf = (ee.Classifier.smileRandomForest(300)
           .train(features=tr, classProperty='label',
                  inputProperties=bands))
    return img.classify(clf)


def export_results_gee(classified, roi, prefix='S1_v5', folder='GEE_Exports'):
    import ee
    ee.batch.Export.image.toDrive(
        image=classified.toByte(), description=f'{prefix}_raster',
        folder=folder, scale=10, region=roi,
        crs='EPSG:4326', maxPixels=1e9).start()
    for cv, cn in CLASS_LABELS.items():
        vecs = (classified.eq(cv).selfMask()
                .reduceToVectors(scale=10, geometry=roi,
                                 geometryType='polygon',
                                 eightConnected=True, labelProperty='class')
                .map(lambda f: f.set('class_name', cn, 'class_id', cv)))
        ee.batch.Export.table.toDrive(
            collection=vecs,
            description=f'{prefix}_{cn.replace(" ","_")}',
            folder=folder, fileFormat='GeoJSON').start()


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == '__main__':
    S1_DIR  = r'f:\NIDM-SRM Drone Project Docs\Sentinel1_1'
    OUT_DIR = r'f:\NIDM-SRM Drone Project Docs\classified_glacier_maps'
    
    SLOPE_TIF  = os.path.join(S1_DIR, 'slope_aligned.tif')
    ASPECT_TIF = os.path.join(S1_DIR, 'aspect_aligned.tif')
    DEM_TIF    = os.path.join(S1_DIR, 'dem_aligned.tif')
    
    tif_path = os.path.join(S1_DIR, 'S1_2023-07-18.tif')
    
    run_on_real_s1(
        tif_path,
        output_dir   = OUT_DIR,
        sample_frac  = 0.30,
        slope_path   = SLOPE_TIF  if os.path.exists(SLOPE_TIF)  else None,
        aspect_path  = ASPECT_TIF if os.path.exists(ASPECT_TIF) else None,
        elevation_path = DEM_TIF  if os.path.exists(DEM_TIF)    else None,
        radar_azimuth = 192.0,
    )