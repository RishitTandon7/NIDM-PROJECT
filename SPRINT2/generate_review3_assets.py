import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd


ROOT = Path(r"f:\NIDM-SRM Drone Project Docs")
OUT_DIR = ROOT / "SPRINT2" / "review3_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_table_image():
    columns = ["Parameter", "Sprint I Result"]
    rows = [
        ["Temporal dataset prepared", "66 Sentinel-2 scenes (2021-2025) with Landsat-5 historical support"],
        ["Core terrain output", "413 micro-watersheds delineated"],
        ["Main technical success", "Automated preprocessing and hydrological mapping"],
        ["Main analytical limitation", "Static thresholding caused false positives in complex glacial terrain"],
        ["Next sprint decision", "Shift to adaptive machine learning classification"],
    ]

    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="center",
        colWidths=[0.28, 0.72],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    for j in range(len(columns)):
        table[0, j].set_facecolor("#174A7E")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        color = "#F4F8FC" if i % 2 == 0 else "white"
        for j in range(len(columns)):
            table[i, j].set_facecolor(color)
            table[i, j].set_edgecolor("#AAB7C4")

    plt.title(
        "Sprint I Outcome Summary",
        fontsize=16,
        fontweight="bold",
        color="#174A7E",
        pad=18,
    )
    plt.tight_layout()
    out = OUT_DIR / "sprint1_result_summary_table.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _box(ax, x, y, w, h, text, fc, ec="#315D7D", fs=12):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)
    return patch


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.5,
            color="#315D7D",
        )
    )


def make_flow_image():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.07, 0.78, 0.22, 0.11, "Sentinel-2 + Landsat-5\nscene collection", "#DCEEFF")
    _box(ax, 0.39, 0.78, 0.22, 0.11, "Preprocessing\nresampling, masking,\nnormalization", "#E8F4E8")
    _box(ax, 0.71, 0.78, 0.22, 0.11, "ASTER GDEM ingestion", "#FFF0D9")

    _box(ax, 0.17, 0.53, 0.25, 0.13, "Optical index generation\nNDWI, NDSI, NDVI", "#F7E6FF")
    _box(ax, 0.58, 0.53, 0.25, 0.13, "Hydrological terrain analysis\n413 micro-watersheds", "#FFE7E7")

    _box(ax, 0.17, 0.28, 0.25, 0.13, "Static threshold classification", "#FFF8CC")
    _box(ax, 0.58, 0.28, 0.25, 0.13, "Sprint I result\nstrong foundation,\nbut weak robustness", "#E6F7F7")

    _box(ax, 0.34, 0.05, 0.32, 0.13, "Key conclusion:\nmove to adaptive ML in Sprint II", "#D7EAFB", fs=13)

    _arrow(ax, 0.29, 0.835, 0.39, 0.835)
    _arrow(ax, 0.61, 0.835, 0.71, 0.835)
    _arrow(ax, 0.50, 0.78, 0.30, 0.66)
    _arrow(ax, 0.82, 0.78, 0.71, 0.66)
    _arrow(ax, 0.295, 0.53, 0.295, 0.41)
    _arrow(ax, 0.705, 0.53, 0.705, 0.41)
    _arrow(ax, 0.295, 0.28, 0.48, 0.18)
    _arrow(ax, 0.705, 0.28, 0.52, 0.18)

    ax.text(
        0.5,
        0.95,
        "Sprint I Outcome and Result Flow",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#174A7E",
    )

    out = OUT_DIR / "sprint1_outcome_flow.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_collage():
    candidates = [
        ROOT / "classified_glacier_maps" / "classified_66_scenes_grid.png",
        ROOT / "S1_classification_map.png",
        ROOT / "internship 2.0" / "final_accurate_maps" / "Final_Classification_Grid_Corrected.png",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None

    fig, axes = plt.subplots(1, len(existing), figsize=(6 * len(existing), 6))
    if len(existing) == 1:
        axes = [axes]

    for ax, path in zip(axes, existing):
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(path.stem.replace("_", " "), fontsize=11, pad=10)

    fig.suptitle(
        "Selected Existing Classification Outputs from the Codebase",
        fontsize=16,
        fontweight="bold",
        color="#174A7E",
        y=0.98,
    )
    plt.tight_layout()
    out = OUT_DIR / "sprint1_existing_results_collage.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_sprint2_table_image():
    columns = ["Parameter", "Sprint II Result"]
    rows = [
        ["Classification approach", "Adaptive machine learning pipeline replacing static threshold-only logic"],
        ["Core analytical output", "Hazard computation completed for 413 watershed networks"],
        ["Exported result tables", "Glacier parameters.xlsx, Glacial lake parameters.xlsx, GLOF Index.xlsx"],
        ["Main technical success", "False positives reduced and outputs serialized into usable relational tables"],
        ["Highest-risk finding", "2 High Hazard targets identified; no Very Severe targets reported"],
    ]

    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="center",
        colWidths=[0.30, 0.70],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    for j in range(len(columns)):
        table[0, j].set_facecolor("#7A2F2F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        color = "#FCF4F4" if i % 2 == 0 else "white"
        for j in range(len(columns)):
            table[i, j].set_facecolor(color)
            table[i, j].set_edgecolor("#BCAAAA")

    plt.title(
        "Sprint II Outcome Summary",
        fontsize=16,
        fontweight="bold",
        color="#7A2F2F",
        pad=18,
    )
    plt.tight_layout()
    out = OUT_DIR / "sprint2_result_summary_table.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_sprint2_hazard_chart():
    categories = ["Very Severe", "High", "Medium", "Low", "Very Low"]
    values = [0, 2, 116, 107, 188]
    colors = ["#7F0000", "#C0392B", "#F39C12", "#5DADE2", "#2ECC71"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, values, color=colors, edgecolor="#2F2F2F", linewidth=0.8)
    ax.set_title("Sprint II Hazard Distribution Across 413 Watersheds", fontsize=16, fontweight="bold", color="#7A2F2F", pad=16)
    ax.set_ylabel("Number of Watersheds / Lakes")
    ax.set_ylim(0, 210)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 4, str(value), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.text(
        0.98,
        0.95,
        "Source: Final_Project_Report.md and\nReview_2_Mega_Document_Extended.txt",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC"),
    )

    plt.tight_layout()
    out = OUT_DIR / "sprint2_hazard_distribution.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_sprint2_flow_image():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.06, 0.78, 0.23, 0.11, "Sprint I baseline outputs\noptical masks + watershed structure", "#E8F0FF")
    _box(ax, 0.38, 0.78, 0.23, 0.11, "Machine learning classification\nadaptive feature weighting", "#FDEDEC")
    _box(ax, 0.70, 0.78, 0.23, 0.11, "Raster-to-vector conversion\nGeoJSON extraction", "#FFF3E0")

    _box(ax, 0.17, 0.53, 0.25, 0.13, "Morphometric parameter extraction\nAAR, SI, CCI, lake geometry", "#EAF7EA")
    _box(ax, 0.58, 0.53, 0.25, 0.13, "Relational flat-file outputs\nGlacier / Lake / GLOF tables", "#F5EEF8")

    _box(ax, 0.17, 0.28, 0.25, 0.13, "Decision logic and\n5-tier hazard scoring", "#FDF2E9")
    _box(ax, 0.58, 0.28, 0.25, 0.13, "Final Sprint II result\n413 networks evaluated", "#E8F8F5")

    _box(ax, 0.34, 0.05, 0.32, 0.13, "Documented distribution:\n0 Very Severe, 2 High, 116 Medium,\n107 Low, 188 Very Low", "#FADBD8", fs=12)

    _arrow(ax, 0.29, 0.835, 0.38, 0.835)
    _arrow(ax, 0.61, 0.835, 0.70, 0.835)
    _arrow(ax, 0.495, 0.78, 0.295, 0.66)
    _arrow(ax, 0.815, 0.78, 0.705, 0.66)
    _arrow(ax, 0.295, 0.53, 0.295, 0.41)
    _arrow(ax, 0.705, 0.53, 0.705, 0.41)
    _arrow(ax, 0.295, 0.28, 0.48, 0.18)
    _arrow(ax, 0.705, 0.28, 0.52, 0.18)

    ax.text(
        0.5,
        0.95,
        "Sprint II Outcome and Result Flow",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#7A2F2F",
    )

    out = OUT_DIR / "sprint2_outcome_flow.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_sprint2_schema_preview():
    glacier_path = ROOT / "ref_for_doc" / "Glacier parameters.xlsx"
    lake_path = ROOT / "ref_for_doc" / "Glacial lake parameters.xlsx"
    glof_path = ROOT / "ref_for_doc" / "GLOF Index.xlsx"

    glacier_df = pd.read_excel(glacier_path).head(3)[["W_ID", "ELA_m", "Accum_Area_sqkm", "Glacier_Area_sqkm", "AAR", "Year"]]
    lake_df = pd.read_excel(lake_path).head(3)[["W_ID", "Lake_Length_m", "Lake_Mean_Width_m", "GLEQ", "Year"]]
    glof_df = pd.read_excel(glof_path).head(3)[["W_ID", "ICI", "SI", "AAR", "CCI", "Year"]]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    datasets = [
        ("Glacier Parameters Preview", glacier_df, "#1F618D"),
        ("Glacial Lake Parameters Preview", lake_df, "#7D6608"),
        ("GLOF Index Preview", glof_df, "#922B21"),
    ]

    for ax, (title, df, color) in zip(axes, datasets):
        ax.axis("off")
        table = ax.table(
            cellText=df.round(4).astype(str).values,
            colLabels=list(df.columns),
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        for j in range(len(df.columns)):
            table[0, j].set_facecolor(color)
            table[0, j].set_text_props(color="white", fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", color=color, pad=8)

    plt.tight_layout()
    out = OUT_DIR / "sprint2_exported_tables_preview.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    outputs = [
        make_table_image(),
        make_flow_image(),
        make_collage(),
        make_sprint2_table_image(),
        make_sprint2_hazard_chart(),
        make_sprint2_flow_image(),
        make_sprint2_schema_preview(),
    ]
    print("Generated assets:")
    for output in outputs:
        if output is not None:
            print(output)


if __name__ == "__main__":
    main()
