import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)

# ── Data ──────────────────────────────────────────────────────────────────────
rows = [
    ("Sprint 1", "Data Acquisition",  "Sentinel-2 & Landsat-5 retrieval via GEE API",         "Rishit",   "Week 1–2",  False),
    ("Sprint 1", "Data Acquisition",  "ASTER GDEM ingestion & D8 sink-filling",                "Koushani", "Week 1–2",  True),
    ("Sprint 2", "Processing",        "NDWI / NDSI / NDVI spectral index computation",          "Rishit",   "Week 3–4",  False),
    ("Sprint 2", "Processing",        "413 Micro-Watershed delineation from ASTER GDEM",       "Koushani", "Week 3–4",  True),
    ("Sprint 3", "ML Classification", "Decision Tree (DT) training & GeoJSON vectorization",   "Rishit",   "Week 5–6",  False),
    ("Sprint 3", "Morphometry",       "AAR · SI · L/W Ratio · CCI parameter computation",      "Koushani", "Week 5–6",  True),
    ("Sprint 4", "Index & Validation","GLOF Susceptibility Index (5-Tier Warning Output)",     "Rishit",   "Week 7–8",  False),
    ("Sprint 4", "Index & Validation","Sikkim Field Survey — NIDM Ground-Truth (Dec 2025)",    "Koushani", "Week 7–8",  True),
    ("Future",   "Phase 2",           "Sentinel-1 SAR cloud-penetration radar integration",    "Both",     "Post Rev 2",False),
    ("Future",   "Phase 3",           "Autonomous Drone UAV survey over High Hazard lakes",    "Both",     "Post Rev 3",True),
]

headers = ["Sprint", "Phase", "Activity", "Assignee", "Timeline"]
col_widths = [1.4, 2.2, 6.0, 1.8, 1.6]
col_x = [0]
for w in col_widths[:-1]:
    col_x.append(col_x[-1] + w)

row_height = 0.45
header_y = 5.5

# ── Header ────────────────────────────────────────────────────────────────────
HEADER_BG  = "#1A5276"
ROW_LIGHT  = "#FFFFFF"
ROW_ALT    = "#D6EAF1"
FUTURE_CLR = "#BDC3C7"
ACCENT     = "#1A5276"

for i, (hdr, w, x) in enumerate(zip(headers, col_widths, col_x)):
    rect = FancyBboxPatch((x+0.03, header_y-0.33), w-0.06, row_height,
                          boxstyle="round,pad=0.04", linewidth=0,
                          facecolor=HEADER_BG)
    ax.add_patch(rect)
    ax.text(x + w/2, header_y - 0.05, hdr,
            ha='center', va='center', fontsize=11,
            fontweight='bold', color='white')

# ── Rows ─────────────────────────────────────────────────────────────────────
for r_idx, row in enumerate(rows):
    y = header_y - (r_idx + 1) * row_height - 0.1
    is_future = row[0] == "Future"
    bg = FUTURE_CLR if is_future else (ROW_ALT if row[5] else ROW_LIGHT)

    for c_idx, (val, w, x) in enumerate(zip(row[:5], col_widths, col_x)):
        rect = FancyBboxPatch((x+0.03, y-0.33), w-0.06, row_height-0.04,
                              boxstyle="round,pad=0.03", linewidth=0.4,
                              edgecolor="#AAB7B8", facecolor=bg)
        ax.add_patch(rect)
        fs = 9 if c_idx == 2 else 10
        fw = 'bold' if c_idx in [0, 3] else 'normal'
        color = "#717D7E" if is_future else "#1B2631"
        ax.text(x + w/2, y - 0.06, val,
                ha='center', va='center', fontsize=fs,
                fontweight=fw, color=color, wrap=True)

# ── Legend ───────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=ROW_LIGHT,  edgecolor='#AAB7B8', label='Rishit'),
    mpatches.Patch(facecolor=ROW_ALT,    edgecolor='#AAB7B8', label='Koushani'),
    mpatches.Patch(facecolor=FUTURE_CLR, edgecolor='#AAB7B8', label='Future Scope'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=9,
          framealpha=0.9, edgecolor='#AAB7B8')

ax.set_title("Work Plan / Timeline — GLOF Vulnerability Classification Pipeline",
             fontsize=13, fontweight='bold', color=ACCENT, pad=10)

plt.tight_layout()
out = r'f:\NIDM-SRM Drone Project Docs\workplan_table.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved:", out)
