import rasterio, numpy as np
from scipy.ndimage import label as ndlabel, gaussian_filter
from skimage.measure import regionprops

tif = r'f:\NIDM-SRM Drone Project Docs\Sentinel1_1\S1_2023-07-18.tif'
with rasterio.open(tif) as src:
    vv = src.read(1).astype(np.float32)
    vh = src.read(2).astype(np.float32)

vv_sm = gaussian_filter(vv, sigma=2.5)
vh_sm = gaussian_filter(vh, sigma=2.5)

lake_raw = (vv_sm < -13.0) & (vh_sm < -22.0)
labeled, n = ndlabel(lake_raw)
sizes = np.bincount(labeled.ravel()); sizes[0] = 0

props = regionprops(labeled)
props_big = [p for p in props if p.area >= 1500]
props_big.sort(key=lambda p: p.area, reverse=True)

print(f"Total raw blobs: {n},  blobs >= 1500px: {len(props_big)}\n")
print(f"{'Rank':>4}  {'Area':>8}  {'Eccent':>7}  {'Solidity':>8}  {'VV_mean':>8}  {'VH_mean':>8}  {'Verdict'}")
print("-"*70)
for i, p in enumerate(props_big[:30]):
    blob = labeled == p.label
    vv_m = float(np.mean(vv[blob]))
    vh_m = float(np.mean(vh[blob]))
    verdict = "LAKE" if (p.eccentricity < 0.97 or p.solidity > 0.30) else "SHADOW"
    print(f"{i+1:>4}  {p.area:>8,}  {p.eccentricity:>7.3f}  {p.solidity:>8.3f}  {vv_m:>8.1f}  {vh_m:>8.1f}  {verdict}")
