import sys
import subprocess

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'python-pptx', '--quiet'])
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import copy

# ─── LOAD USER'S TEMPLATE ────────────────────────────────────────────────────
prs = Presentation(r'c:\Users\Rishit\Downloads\Minor Project_Ctech PPT Template (2).pptx')

# Helper: get slide layout by index
TITLE_CONTENT_LAYOUT = prs.slide_layouts[1]   # Title + Content
TITLE_ONLY_LAYOUT    = prs.slide_layouts[5]   # Title Only (for image slides)
BLANK_LAYOUT         = prs.slide_layouts[6]   # Blank

def add_slide(title_text, bullets=None, sub_heading=None):
    """Add a standard title+bullets slide."""
    slide = prs.slides.add_slide(TITLE_CONTENT_LAYOUT)
    # Title
    title = slide.shapes.title
    title.text = title_text
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.size = Pt(28)

    if bullets:
        tf = slide.placeholders[1].text_frame
        tf.clear()
        for i, b in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = b
            p.font.size = Pt(16)
            p.level = 0
    return slide

def add_image_placeholder_slide(title_text, caption):
    """Image placeholder slide (title only + text box for caption)."""
    slide = prs.slides.add_slide(TITLE_ONLY_LAYOUT)
    title = slide.shapes.title
    title.text = title_text
    title.text_frame.paragraphs[0].font.size = Pt(24)
    title.text_frame.paragraphs[0].font.bold = True

    # Caption box in center
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(1.8), Inches(9), Inches(0.6))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = caption
    p.font.size = Pt(13)
    p.font.italic = True
    p.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
    p.alignment = PP_ALIGN.CENTER

    # Big placeholder box
    box = slide.shapes.add_textbox(Inches(1), Inches(2.5), Inches(8), Inches(3.5))
    tf2 = box.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "[ INSERT IMAGE HERE ]"
    p2.font.size = Pt(20)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(0xCC, 0x44, 0x44)
    p2.alignment = PP_ALIGN.CENTER
    return slide

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1: TITLE SLIDE (use existing blank first slide from template)
# ══════════════════════════════════════════════════════════════════════════════
title_slide = prs.slides[0]
for ph in title_slide.placeholders:
    if ph.placeholder_format.idx == 0:
        ph.text = "Automated GLOF Vulnerability Classification\nand Warning Generation Pipeline"
        ph.text_frame.paragraphs[0].font.size = Pt(28)
        ph.text_frame.paragraphs[0].font.bold = True
    elif ph.placeholder_format.idx == 1:
        ph.text = "Rishit Tandon & Koushani  |  NIDM-Funded GLOF Study  |  Review 2"
        ph.text_frame.paragraphs[0].font.size = Pt(16)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A: PROBLEM STATEMENT
# ══════════════════════════════════════════════════════════════════════════════
add_slide(
    "Problem Statement",
    [
        "Glacial Lake Outburst Floods (GLOFs) pose severe hydrological risks in High Mountain Asia (HMA).",
        "Rapid glacial retreat accumulates meltwater behind loose, unconsolidated terminal moraines.",
        "These moraine barriers lack the structural integrity to sustain increasing hydrostatic pressure.",
        "Failure of these dams precipitates catastrophic downstream flooding, threatening settlements and infrastructure.",
        "Traditional monitoring involves laborious manual GIS digitization — slow and unfeasible for early warning.",
        "High-frequency physical monitoring is impossible given the remoteness and structural instability of terrain.",
    ]
)

add_slide(
    "Understanding the Problem: What is a GLOF?",
    [
        "Glacier = A massive body of slow-moving ice formed from accumulated snowfall over centuries.",
        "As climate changes, glaciers melt → creating large meltwater lakes trapped behind natural earth dams called moraines.",
        "Moraine = An unstable wall of loose rocks, ice and soil deposited by glacial movement.",
        "Moraine Dam Failure = Hydrostatic pressure builds until the dam collapses suddenly.",
        "The resulting GLOF releases millions of cubic meters of water downstream in minutes.",
        "Chungthang (Sikkim, 2023): A real GLOF catastrophe that our team personally observed during our field visit.",
    ]
)

add_slide(
    "Why Automated Remote Sensing?",
    [
        "Panel Q: Why not just send people to check on the lakes?",
        "→ These lakes are at 4,000–6,000m altitude. No roads. No mobile signal. Extreme weather.",
        "Panel Q: Why not use normal cameras / drones already?",
        "→ The lakes span hundreds of km². A satellite covers 10,000 km² in a single image.",
        "Panel Q: So why can't we just look at the satellite images manually?",
        "→ 66 images × multiple pixel layers = millions of data points. Only algorithms can process this at scale.",
        "Solution: Automate detection using Python + Machine Learning over multi-decadal satellite archives.",
    ]
)

add_slide(
    "Problem Statement: Visual Summary",
    [
        "[ IMAGE SLIDE — see next slide ]",
    ]
)
add_image_placeholder_slide(
    "Field Evidence: The 2023 Chungthang GLOF Aftermath",
    "Insert your best field visit photograph showing the riverbank erosion / disaster damage from the 2023 Chungthang GLOF event."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B: OBJECTIVES
# ══════════════════════════════════════════════════════════════════════════════
add_slide(
    "Research Objectives",
    [
        "1. Integrate multi-spectral remote sensing with automated machine learning classification.",
        "2. Utilize Sentinel-2 (10m) and Landsat-5 (30m) via Google Earth Engine (1988–2025).",
        "3. Delineate 413 independent Micro-Watersheds using ASTER GDEM flow-accumulation modeling.",
        "4. Deploy a supervised Decision Tree (DT) classifier for Glacier / Lake / Land segmentation.",
        "5. Compute morphometric vulnerability parameters: AAR, Snout Index, L/W Ratio, CCI.",
        "6. Generate an operationalized 5-Tier GLOF Susceptibility Index for 413 watersheds.",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C: DATA ACQUISITION
# ══════════════════════════════════════════════════════════════════════════════
add_slide(
    "Data Acquisition: What Data Did We Use?",
    [
        "Dataset 1: Sentinel-2 (MSI) — 10m resolution — 2021 to 2025 — via Google Earth Engine.",
        "Dataset 2: Landsat-5 (TM) — 30m resolution — 1988 to 2011 — Historical baseline.",
        "Dataset 3: ASTER GDEM — 30m Digital Elevation Model — NASA/METI — Topographic analysis.",
        "Total: 66 individual satellite scenes processed across the 35-year temporal window.",
        "All satellite data was fetched programmatically using the GEE Python API — zero manual downloading.",
    ]
)

add_slide(
    "Understanding the Data: What is a Satellite Image?",
    [
        "A satellite image is NOT one photograph. It is a stack of invisible electromagnetic wavelength layers called 'Bands'.",
        "Band 3 (Green light) + Band 8 (Near-Infrared) = Isolate water → NDWI calculation.",
        "Band 3 (Green) + Band 11 (Short-Wave Infrared) = Isolate snow/ice → NDSI calculation.",
        "ASTER GDEM = A 3D terrain elevation map (like a topography scan) used to trace water flow downhill.",
        "Panel Q: Where does the data come from? → Copernicus / USGS / NASA satellites orbiting 786km above Earth.",
        "Panel Q: How often does the satellite take pictures? → Sentinel-2: Every 5 days. Sentinel-1 (SAR): Every 12 days.",
    ]
)

add_slide(
    "Hydrological Processing: 413 Micro-Watersheds",
    [
        "ASTER GDEM elevation data was processed using the D8 deterministic flow-accumulation algorithm.",
        "Step 1: Fill Sinks — removes local depression errors in the DEM to allow water to flow.",
        "Step 2: Flow Direction — computes the mathematical direction water travels from each pixel.",
        "Step 3: Flow Accumulation — traces how many upstream cells drain into each downstream cell.",
        "Step 4: Pour Point Extraction — glacial lake locations are used as outlets to delineate catchment zones.",
        "Result: 413 independent Micro-Watershed boundaries extracted and vectorized as GeoJSON polygons.",
    ]
)

add_image_placeholder_slide(
    "Data Output: 1988–2023 Individual Band and FCC Grids",
    "Insert the 'Individual Band' and 'FCC' grids from your 1988-2023.docx file here. This proves the multi-decadal dataset retrieval."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION D: IMAGE PROCESSING & ML CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
add_slide(
    "Image Processing: Spectral Index Computation",
    [
        "NDWI = (Green − NIR) / (Green + NIR) → Isolates liquid water bodies.",
        "NDSI = (Green − SWIR) / (Green + SWIR) → Isolates glacial snow and ice.",
        "NDVI = (NIR − Red) / (NIR + Red) → Maps vegetation to eliminate false positives.",
        "Outputs: A classified pixel matrix where each pixel is tagged: Water / Ice / Vegetation / Terrain.",
        "Problem: Topological shadows in mountainous terrain mimic water reflectance → False Positives.",
        "Solution: Upgrade from static index thresholding to a Machine Learning model.",
    ]
)

add_slide(
    "Machine Learning: Decision Tree Classifier",
    [
        "Panel Q: What is a Decision Tree?",
        "→ Think of it as a flowchart of Yes/No questions applied to each pixel's spectral values.",
        "Example: Is NDWI > 0.0? → Yes → Is NDSI < 0.4? → Yes → GLACIAL LAKE.",
        "Training: The model was trained on manually verified ground-truth pixels from Dec 2021 scene.",
        "Output: 3-Class Segmentation: Glacier (Blue) / Glacial Lake (Cyan) / Non-Glacier (Grey).",
        "Key Advantage: Eliminates shadow false-positives that fooled standard NDWI thresholding.",
        "Post-processing: Pixel boundaries polygonized into GeoJSON vector coordinates.",
    ]
)

add_image_placeholder_slide(
    "Classification Output: 66-Scene Sentinel-2 Batch Grid",
    "Insert your large 'Raster' 66-scene batch classified output here. Proves the ML pipeline processed all temporal scenes."
)

add_image_placeholder_slide(
    "Classification Output: Vector Segmentation (QGIS Boundaries)",
    "Insert your 'Vector Segmentation' QGIS boundary maps here. Shows the precise GPS coordinates extracted per glacier and lake."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION E: MORPHOMETRIC COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
add_slide(
    "Morphometric Parameter Computation",
    [
        "Once the spatial boundaries are mapped, the physical threat must be quantified using geomorphic equations.",
        "3 Categories of Parameters: Glacier | Glacial Lake | Catchment",
        "Glacier: Accumulation Area Ratio (AAR) and Snout Index (SI).",
        "Glacial Lake: Lake Length-to-Width (L/W) Ratio.",
        "Catchment: Catchment Compactness Index (CCI).",
        "All parameters computed programmatically for all 413 watersheds and serialized to Excel files.",
    ]
)

add_slide(
    "Category 1: Glacier Parameters",
    [
        "Accumulation Area Ratio (AAR) = Accumulation Zone Area / Total Glacier Area",
        "→ AAR < 0.5 = Glacier is in terminal ablation phase (melting faster than accumulation).",
        "→ Rapidly declining AAR over 35 years proves irreversible glacial retreat.",
        "Snout Index (SI) = GDEM slope angle at the glacial terminus (snout base).",
        "→ A steep snout increases the kinetic energy of potential ice-calving events into the lake.",
        "Panel Q: Why does snout steepness matter?",
        "→ A steep glacier face drops ice blocks with much greater force into the lake, creating displacement waves.",
    ]
)

add_image_placeholder_slide(
    "Glacier Morphometric Output",
    "Insert the 'Module 2: Glacier Morphometric Computation' color maps here showing the glacial decay and AAR distribution."
)

add_slide(
    "Category 2: Glacial Lake Parameters",
    [
        "Lake Length-to-Width (L/W) Ratio = Lake Length / Mean Width",
        "→ Elongated lakes (High L/W ratio) are significantly more dangerous during outburst events.",
        "→ A long, narrow lake acts as a velocity funnel, amplifying the displacement wave before it hits the dam.",
        "Panel Q: Why is the shape of the lake important?",
        "→ A circular lake absorbs displacement waves from all sides evenly.",
        "→ A long, thin lake channels the energy in one direction — directly into the moraine dam.",
    ]
)

add_image_placeholder_slide(
    "Glacial Lake Morphometric Output",
    "Insert the 'Module 3: Glacial Lake Morphometric Computation' maps here characterizing the lake boundary dimensions."
)

add_slide(
    "Category 3: Catchment Parameters",
    [
        "Catchment Compactness Index (CCI) = 0.2821 × (Perimeter / √(Area))",
        "→ Measures the geometric 'roundness' of the watershed basin.",
        "→ A perfectly circular basin = CCI of 1.0 = maximum hydrostatic danger.",
        "Panel Q: Why does the shape of the surrounding valley matter?",
        "→ In a circular basin, all rainfall funnels simultaneously from 360° directly into the lake.",
        "→ This creates sudden, unpredictable hydrostatic pressure spikes — far more dangerous than elongated basins.",
        "Calculated for all 413 micro-watersheds extracted from the ASTER GDEM.",
    ]
)

add_image_placeholder_slide(
    "Catchment Morphometric Output: Ice Compactness Index (ICI)",
    "Insert the ICI colored 413 watershed maps from Morphometry 2021.docx. Shows the Red Zone (Very High CCI) distribution."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION F: GLOF INDEX
# ══════════════════════════════════════════════════════════════════════════════
add_slide(
    "GLOF Index: Final Susceptibility Assessment",
    [
        "All morphometric features (AAR, SI, L/W Ratio, CCI) are synthesized into the Decision Tree.",
        "The model evaluates the combined feature weights and outputs a localized threat score.",
        "5-Tier Susceptibility Classification:",
        "   → Very High Hazard: 2 Micro-Watersheds",
        "   → Medium Hazard: 116 Micro-Watersheds",
        "   → Low Hazard: 107 Micro-Watersheds",
        "   → Very Low Hazard: 188 Micro-Watersheds",
        "Output serialized to GLOF_Index.xlsx — ready for disaster management deployment.",
    ]
)

add_slide(
    "Understanding the GLOF Index",
    [
        "Panel Q: What does 'Very High Hazard' mean exactly?",
        "→ It means that specific valley has the worst combination of: steep glacier + elongated lake + compact catchment.",
        "Panel Q: What do officials do with this information?",
        "→ Pre-position rescue teams. Build early-warning sirens. Restrict downstream settlements.",
        "Panel Q: How accurate is your prediction?",
        "→ The field visit to Chungthang (2023 GLOF site) confirmed our algorithm's high-risk flagging was correct.",
        "→ The real-world geomorphic destruction we observed in North Sikkim matched the theoretical CCI scores exactly.",
    ]
)

add_image_placeholder_slide(
    "GLOF Index Output Map",
    "Insert the correct finalized 'Module 5: GLOF INDEX' threat probability map here. Use the 2nd correct output ONLY."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION G: ANALYSIS & RESULTS
# ══════════════════════════════════════════════════════════════════════════════
add_slide(
    "Analysis: Morphometric Correlation",
    [
        "Key Finding 1: Rapid decadal AAR decline directly correlates with physical lake expansion.",
        "→ As glacier ablates (loses mass), meltwater accelerates ice-core moraine degradation.",
        "→ This acts as a positive feedback loop: smaller glacier → weaker dam → bigger lake → higher risk.",
        "Key Finding 2: Lakes with high L/W Ratio situated below high Snout Index glaciers showed maximum vulnerability.",
        "→ Steep calving glacier + elongated lake = exponentially amplified displacement wave energy.",
        "Key Finding 3: CCI values positively correlate with sudden post-rainfall lake level surges.",
    ]
)

add_slide(
    "Results: Data Output Summary",
    [
        "Glacier_Parameters.xlsx → AAR, Snout Index computed for all 413 watersheds.",
        "Glacial_Lake_Parameters.xlsx → Lake Length, Mean Width, L/W Ratio per watershed.",
        "GLOF_Index.xlsx → CCI, ICI, SI, AAR combined into the final 5-tier warning score.",
        "GeoJSON Vector Files → Precise GPS lake/glacier boundaries for GIS deployment.",
        "66-Scene PNG Grid → Full visual temporal progression (1988–2023) of glacial retreat.",
        "All outputs generated programmatically using Python (Pandas, Rasterio, NumPy, Geopandas).",
    ]
)

add_image_placeholder_slide(
    "Data Output: GLOF_Index.xlsx Sample",
    "Insert a screenshot of the first 5 rows of your GLOF_Index.xlsx here proving the CCI / SI / AAR / ICI serialization."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION H: FIELD VISIT
# ══════════════════════════════════════════════════════════════════════════════
add_slide(
    "Ground-Truth Validation: Sikkim Field Survey (Dec 2025)",
    [
        "Survey Area: East and North Sikkim, Tista River Basin.",
        "Survey Dates: 19–28 December 2025.",
        "Funded by: NIDM (National Institute of Disaster Management).",
        "Sites Covered: Tsomgo Lake, Kupup Lake, Chungthang (2023 GLOF site), Lachung Valley, Yumthang Valley.",
        "Key Observation: Real-world riverbank erosion after 2023 Chungthang GLOF event observed directly.",
        "→ This empirically validates the 'Very High' CCI threshold produced by our algorithm.",
    ]
)

add_slide(
    "Field Survey: What We Observed and Validated",
    [
        "Observation 1: Frozen lake surfaces blend with surrounding snowfields in winter.",
        "→ Validates why NDSI (snow index) is essential — NDWI alone would miss these lakes entirely.",
        "Observation 2: Temporary natural dam with high-slope conditions observed downstream in East Sikkim.",
        "→ Directly mirrors the High Snout Index (SI) gradient flagged by our algorithm for nearby watersheds.",
        "Target A for Drone Phase: Huang Valley (Hangu, Tsomgo, Sarathang, Kupup Lakes).",
        "Target B for Drone Phase: Yumthang–Lachung Valley (Sanglaphu Cho Lake).",
    ]
)

add_image_placeholder_slide(
    "Field Evidence: Koushani at Survey Site",
    "Insert Koushani's field photo here. IMPORTANT: Write the exact Year (December 2025) in the caption below the image."
)
add_image_placeholder_slide(
    "Field Evidence: Site Photographs",
    "Insert 2–3 additional photographs from your field visit (frozen lakes, erosion damage, team at site)."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION I: FUTURE SCOPE
# ══════════════════════════════════════════════════════════════════════════════
add_slide(
    "Future Scope",
    [
        "Phase 1 (Current): Optical Remote Sensing — Sentinel-2 + Landsat (blocked by clouds during monsoon).",
        "Phase 2 (Next): Sentinel-1 SAR Integration — Active microwave radar penetrates clouds.",
        "→ Sentinel-1 revisits every 12 days — fills the 'missing flood-time days' of Sentinel-2.",
        "→ Critical for detecting rapid lake-level surges during monsoon events when optical view is blocked.",
        "Phase 3 (Future): Autonomous Drone (UAV) Deployment over Very High Risk Lakes.",
        "→ Sub-centimeter 3D moraine mesh models replacing the 10m satellite pixel limit.",
        "→ Priority sites already identified during Dec 2025 field survey (Huang Valley, Sanglaphu Cho Lake).",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION J: CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
add_slide(
    "Conclusion",
    [
        "The methodology validates the viability of autonomous disaster management analytical pipelines.",
        "66 Sentinel-2 scenes + historical Landsat-5 arrays successfully analyzed glacial erosion patterns.",
        "413 micro-watersheds delineated from ASTER GDEM hydrological processing.",
        "Supervised Decision Tree modeling transitions raw satellite imagery into structured Warning Guidelines.",
        "Structural morphometry (CCI, Snout Index, AAR) quantifies physics of the outburst threat precisely.",
        "Framework is scalable, repeatable, and validated by real-world field reconnaissance (2025).",
    ]
)

add_slide(
    "References",
    [
        "1. Copernicus Sentinel-2 Multispectral Instrument (MSI) Data Archives.",
        "2. USGS Landsat-5 Thematic Mapper historical repository (1988–2011).",
        "3. NASA/METI ASTER Global Digital Elevation Model (GDEM).",
        "4. Google Earth Engine (GEE) Python API Documentation.",
        "5. NIDM-Funded GLOF Study Field Visit Report — December 2025.",
    ]
)

# ─── SAVE ─────────────────────────────────────────────────────────────────────
OUTPUT_PATH = r'f:\NIDM-SRM Drone Project Docs\Review_2_Final_PPT.pptx'
prs.save(OUTPUT_PATH)
print(f"SUCCESS: PPT saved to {OUTPUT_PATH}")
print(f"Total slides: {len(prs.slides)}")
