import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy.ndimage import label, sum as ndi_sum


ROOT = Path(r"f:\NIDM-SRM Drone Project Docs")
DEFAULT_TIFF = ROOT / "internship 2.0" / "Sentinel 2" / "Sentinel2_2021-11-08.tif"
DEFAULT_OUT = ROOT / "SPRINT2" / "backend_process_outputs"


def percentile_stretch(arr, p_low=2, p_high=98):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(valid, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    stretched = np.clip((arr - lo) / (hi - lo), 0, 1)
    return stretched.astype(np.float32)


def remove_speckles(mask, min_size):
    labeled, num = label(mask)
    if num == 0:
        return mask
    sizes = ndi_sum(mask, labeled, range(num + 1))
    keep = sizes >= min_size
    return keep[labeled]


def load_scene(tiff_path):
    with rasterio.open(tiff_path) as src:
        blue = src.read(1).astype(np.float32)   # B2
        green = src.read(2).astype(np.float32)  # B3
        red = src.read(3).astype(np.float32)    # B4
        nir = src.read(7).astype(np.float32)    # B8
        swir = src.read(9).astype(np.float32)   # B11
        profile = src.profile
        transform = src.transform
        crs = src.crs
    return blue, green, red, nir, swir, profile, transform, crs


def compute_pipeline(blue, green, red, nir, swir):
    np.seterr(divide="ignore", invalid="ignore")

    ndwi = (green - nir) / (green + nir + 1e-6)
    ndsi = (green - swir) / (green + swir + 1e-6)
    ndvi = (nir - red) / (nir + red + 1e-6)
    avg_vis = (blue + green + red) / 3.0

    raw_lake = (ndwi > 0.1) & (nir < 4500)
    shadow_mask = (avg_vis < 3000) & (ndwi < 0.08)
    lake_after_shadow = raw_lake & (~shadow_mask)
    glacier_raw = (ndsi > 0.0) & (avg_vis > 4000)
    glacier_no_overlap = glacier_raw & (~lake_after_shadow)

    lake_clean = remove_speckles(lake_after_shadow, min_size=8)
    glacier_clean = remove_speckles(glacier_no_overlap, min_size=10)
    nonglacier = (~lake_clean) & (~glacier_clean)

    final_map = np.zeros_like(blue, dtype=np.uint8)
    final_map[glacier_clean] = 1
    final_map[lake_clean] = 2

    return {
        "ndwi": np.nan_to_num(ndwi, nan=0.0),
        "ndsi": np.nan_to_num(ndsi, nan=0.0),
        "ndvi": np.nan_to_num(ndvi, nan=0.0),
        "avg_vis": avg_vis,
        "raw_lake": raw_lake,
        "shadow_mask": shadow_mask,
        "lake_after_shadow": lake_after_shadow,
        "glacier_raw": glacier_raw,
        "glacier_no_overlap": glacier_no_overlap,
        "lake_clean": lake_clean,
        "glacier_clean": glacier_clean,
        "nonglacier": nonglacier,
        "final_map": final_map,
    }


def count_objects(mask, transform):
    count = 0
    pixel_counts = []
    for geom, val in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform):
        if val == 1:
            count += 1
            coords = geom.get("coordinates", [])
            if coords:
                pixel_counts.append(len(coords[0]))
    return count, pixel_counts


def build_figure(scene_name, blue, green, red, nir, swir, results, output_png):
    rgb = np.dstack([
        percentile_stretch(red),
        percentile_stretch(green),
        percentile_stretch(blue),
    ])

    cmap_final = ListedColormap(["#8B4513", "#D3D3D3", "#00BFFF"])

    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    axes = axes.flatten()

    panels = [
        (rgb, f"{scene_name}\nRGB Composite", None),
        (percentile_stretch(nir), "NIR (B8)", "gray"),
        (percentile_stretch(swir), "SWIR (B11)", "gray"),
        (percentile_stretch(results["avg_vis"]), "Average Visible Brightness", "gray"),
        (results["ndwi"], "NDWI", "Blues"),
        (results["ndsi"], "NDSI", "PuBuGn"),
        (results["ndvi"], "NDVI", "YlGn"),
        (results["raw_lake"], "Raw Lake Mask\nNDWI > 0.1 and NIR < 4500", "Blues"),
        (results["shadow_mask"], "Shadow Mask\nAvgVIS < 3000 and NDWI < 0.08", "Greys"),
        (results["lake_after_shadow"], "Lake After Shadow Removal", "Blues"),
        (results["glacier_clean"], "Final Glacier Mask", "binary"),
        (results["final_map"], "Final Classification\nBrown=Non-Glacier Grey=Glacier Blue=Lake", cmap_final),
    ]

    for ax, (data, title, cmap) in zip(axes, panels):
        if data.ndim == 3:
            ax.imshow(data)
        else:
            ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle("Backend Process of Image Classification", fontsize=20, fontweight="bold", color="#174A7E")
    plt.tight_layout()
    plt.savefig(output_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_report(scene_name, results, transform, output_txt):
    lake_pixels_raw = int(results["raw_lake"].sum())
    lake_pixels_final = int(results["lake_clean"].sum())
    glacier_pixels_final = int(results["glacier_clean"].sum())
    shadow_pixels = int(results["shadow_mask"].sum())
    nonglacier_pixels = int(results["nonglacier"].sum())

    lake_objects, _ = count_objects(results["lake_clean"], transform)
    glacier_objects, _ = count_objects(results["glacier_clean"], transform)

    report = f"""Scene: {scene_name}

Pipeline stages:
1. Load Sentinel-2 bands (B2, B3, B4, B8, B11)
2. Compute NDWI, NDSI, NDVI, and average visible brightness
3. Detect raw lake candidates using NDWI and NIR thresholding
4. Suppress shadow-driven false positives
5. Detect glacier candidates using NDSI and brightness
6. Remove overlaps and small speckle regions
7. Produce final class map

Pixel statistics:
- Raw lake candidate pixels: {lake_pixels_raw}
- Shadow pixels filtered: {shadow_pixels}
- Final lake pixels: {lake_pixels_final}
- Final glacier pixels: {glacier_pixels_final}
- Final non-glacier pixels: {nonglacier_pixels}

Object statistics:
- Final lake polygons/components: {lake_objects}
- Final glacier polygons/components: {glacier_objects}

Class labels:
- 0: Non-glacier / background
- 1: Glacier
- 2: Glacier lake
"""

    output_txt.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Visualize the full backend image-classification process.")
    parser.add_argument("--tiff", default=str(DEFAULT_TIFF), help="Path to the input TIFF scene.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUT), help="Directory for output files.")
    args = parser.parse_args()

    tiff_path = Path(args.tiff)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scene_name = tiff_path.stem
    output_png = outdir / f"{scene_name}_backend_process.png"
    output_txt = outdir / f"{scene_name}_backend_process_report.txt"

    blue, green, red, nir, swir, profile, transform, crs = load_scene(tiff_path)
    results = compute_pipeline(blue, green, red, nir, swir)

    build_figure(scene_name, blue, green, red, nir, swir, results, output_png)
    build_report(scene_name, results, transform, output_txt)

    print(f"Saved backend visualization: {output_png}")
    print(f"Saved backend report: {output_txt}")


if __name__ == "__main__":
    main()
