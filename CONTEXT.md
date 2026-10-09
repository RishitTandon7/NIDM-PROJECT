# PROJECT CONTEXT & COMPREHENSIVE REPOSITORY BLUEPRINT
**NIDM-SRM Drone & Remote Sensing GLOF Early Warning System**

---

## 1. Executive Summary & Administrative Context

### 1.1 Project Overview
This repository hosts the research, computational algorithms, geospatial datasets, field study findings, and hardware architecture for the disaster mitigation initiative funded by the **National Institute of Disaster Management (NIDM)**, Ministry of Home Affairs, Government of India, executed in collaboration with **SRM Institute of Science and Technology (SRMIST)**.

* **Primary Project Title:** *Development of a GLOF Warning System through Strong Endurance Drone-based Long-Range (LoRa) Communication for the Upper Catchment of the Tista River*
* **Grant & Sponsorship:** NIDM Research Project on Disaster Management: Innovation & Action Research
* **Project Code:** `RP24` (Sanctioned Grant: INR 8,00,000 / 8 Lakhs, 24-Month Duration)
* **Host Institution:** Department of Computing Technologies (CTech), School of Computing, SRM Institute of Science and Technology, Kattankulathur, Chennai, India
* **Principal Investigator (PI):** Dr. Manoj Kumar Rana
* **Co-Principal Investigator (Co-PI):** Dr. Snehasish Ghosh
* **Student Researchers / Interns:** Rishit Tandon & Koushani
* **Geographic Research Area:** Upper Tista River Basin, North & East Sikkim Himalayas (High Mountain Asia), including South Lhonak Lake, Tsomgo Lake, Kupup Lake, and the downstream Chungthang disaster axis.

---

## 2. Multi-Tier System Architecture (The Big Picture)

The initiative solves the catastrophic threat of **Glacial Lake Outburst Floods (GLOFs)**—such as the devastating October 2023 South Lhonak lake breach and Chungthang dam destruction in Sikkim—by deploying a synchronized **Three-Tier Early Warning System**:

```
+---------------------------------------------------------------------------------------------------+
|                                     TIER 1: MACRO OBSERVATION                                     |
|  Orbital Remote Sensing: Sentinel-2 Optical (10m) + Sentinel-1 SAR Radar + Landsat-5 Baselines     |
|  * 413 Micro-Watersheds Delineated (ASTER GDEM 30m)                                               |
|  * ML Classification (Decision Tree / Random Forest) -> 5-Tier GLOF Hazard Index                  |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v (Flags "High" & "Very High" Risk Zones)
+---------------------------------------------------------------------------------------------------+
|                                  TIER 2: TACTICAL RECONNAISSANCE                                  |
|  High-Endurance Autonomous Fixed-Wing UAV (Drone) Deployment                                      |
|  * Long-Range LoRa Telemetry + 4G/5G Cellular Payload Transmission                                |
|  * Centimeter-Level RTK GNSS (u-blox F9P / Here+) + Pixhawk / Cube Orange Autopilot               |
|  * Sub-Centimeter Photogrammetry & LIDAR for Moraine Structural Integrity Monitoring              |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v (Critical Threshold Trigger)
+---------------------------------------------------------------------------------------------------+
|                                  TIER 3: EARLY WARNING DISSEMINATION                              |
|  Automated Threat Serialization & Municipal Notification Engine                                   |
|  * GeoJSON Polygons + Flat-File Relational Spreadsheets (W_ID Primary Key)                        |
|  * Real-time automated alerts to SDMA, NDMA, District Collectors, and Hydroelectric Stations     |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. The Minor Project Component (`21CSP302L`)

### 3.1 Course Details & Purpose
* **Course Code:** `21CSP302L - Minor Project`
* **Department:** Department of Computing Technologies (CTech), School of Computing, SRMIST
* **Project Title:** *Automated Glacial Lake Outburst Flood (GLOF) Vulnerability Classification and Warning Generation Pipeline*
* **Student Team:** Rishit Tandon & Koushani
* **Scope:** The Minor Project directly delivered **Tier 1 and Tier 3** of the overarching NIDM research mandate—architecting, implementing, and validating the **fully automated satellite data processing, machine learning classification, morphometric risk indexing, and relational threat serialization software pipeline**.

### 3.2 Agile Sprints & Execution Milestones

#### Sprint I: Hydrological Preprocessing & Optical Feature Engineering
* **Data Acquisition:** Integrated Google Earth Engine (GEE) Python API workflows to ingest **66 batch scenes of Copernicus Sentinel-2** (2021–2025) and legacy **Landsat-5 TM** (1988–2011) archives to quantify 35-year decadal retreat baselines.
* **Hydrological Extraction:** Processed 30-meter NASA/METI ASTER Global Digital Elevation Models (GDEM) with sink-filling and D8 deterministic flow-direction/accumulation algorithms. Autonomous programmatic extraction identified **413 distinct Micro-Watersheds (MWs)** across the Upper Tista Basin without manual GIS digitizing.
* **Initial Challenge Identified:** Single-factor boolean thresholding (e.g., `NDWI > 0.3`) suffered up to an 18% false-positive rate due to high-altitude mountain valley shadows and turbid meltwater spectral overlap.

#### Sprint II: Machine Learning Classification & Morphometric Risk Modeling
* **Supervised Machine Learning:** Replaced hard thresholding with an optimized **Decision Tree (DT)** classifier (supported by Random Forest and SVM benchmarks in Python). Sliced multi-band feature stacks (NDWI, NDSI, NDVI, DEM Elevation, Slope) into clean 3-class segmentation:
  * `Class 0`: Non-Glacier / Barren Rock / Shadow
  * `Class 1`: Glacier / Ice / Snow
  * `Class 2`: Glacial Lake / Open Water
* **Polygon Vectorization:** Developed vectorized raster-to-polygon routines (`rasterio.features.shapes`, `shapely`, `geopandas`) to convert raster classification masks into closed-loop **GeoJSON vector polygons** with real-world geographic coordinates.
* **Morphometric Parameter Computation:** Algorithmically calculated physical vulnerability variables per micro-watershed:
  * **Glacier Decay:** Accumulation Area Ratio ($AAR = \text{Accumulation Area} / \text{Total Area}$) where $AAR < 0.5$ denotes terminal retreat; Snout Index ($SI$) measuring terminus slope gradient.
  * **Lake Geometry:** Length-to-Width Ratio ($L/W$) measuring wave amplification and water funneling dynamics.
  * **Catchment Shape:** Catchment Compactness Index ($CCI = 0.2821 \times \text{Perimeter} / \sqrt{\text{Area}}$), where circular basins ($CCI \to 1.0$) indicate rapid hydrostatic runoff accumulation.
* **Relational Flat-File Database Architecture:** Unified spatial data via primary key **`W_ID`** (Micro-Watershed ID: 1 to 413) across:
  * `Glacier_parameters.xlsx`
  * `Glacial_lake_parameters.xlsx`
  * `GLOF_Index.xlsx`
* **5-Tier GLOF Susceptibility Distribution:**
  * **Very High / High Hazard:** 2 Micro-watersheds (South Lhonak & adjacent catastrophic trigger zones)
  * **Medium Hazard:** 116 Micro-watersheds
  * **Low Hazard:** 107 Micro-watersheds
  * **Very Low Hazard:** 188 Micro-watersheds

#### Sprint III & Academic Presentations:
* **Review 1 & Review 2 Presentations:** Completed official academic evaluations using customized PPT decks (`Review_2_Final_PPT.pptx` generated via `generate_ppt.py`, based on the departmental `Minor Project_Ctech PPT Template`).
* **Technical Documentation:** Generated comprehensive review literature, including `Review_2_Mega_Document.md`, `Architecture_Document.md`, and `Final_Project_Report.md`.

---

## 4. Ground-Truth Terrestrial Field Validation (December 2025)

To validate theoretical and algorithmic outputs against reality, a field expedition was undertaken:
* **Location:** East & North Sikkim, Upper Tista River Basin
* **Survey Dates:** 19–28 December 2025
* **Key Sites Inspected:** Tsomgo Lake, Kupup Lake, Chungthang (2023 GLOF epicenter & hydro dam breach site), Lachung valley.
* **Empirical Observations & Validation:**
  1. **Winter Ice Albedo:** Confirmed that winter ice surfaces visually and spectrally blend with surrounding snowfields, validating why multi-index triangulation ($NDSI + NDWI$) is mandatory.
  2. **Morphometric Accuracy:** Observed extreme riverbank scouring, catastrophic vertical erosion, and severe valley wall collapse at Chungthang that directly mirrored the extreme risk parameters computed by the $CCI$ and $SI$ models for that drainage corridor.

---

## 5. Drone (UAV) Hardware & Telemetry Setup (Tier 2)

As part of the NIDM grant scope (`Drone Setup/Drone Project- RTF - NCO4G.xlsx`), the tactical aerial reconnaissance platform is specified as follows:

| System / Category | Specified Hardware Component | Purpose / Operational Specification |
| :--- | :--- | :--- |
| **Airframe Materials** | 3mm Birch Wood Sheets, Foamboard, Carbon Fiber Spars (2m) | Custom high-endurance, lightweight fixed-wing aerial frame |
| **Autopilot / FC** | Cube Orange / Pixhawk 6X | Redundant IMU, automated waypoint navigation in high-altitude winds |
| **GNSS / RTK** | u-blox F9P / Here+ RTK Base & Air Module | Centimeter-level positioning for precise photogrammetric survey |
| **Long-Range RC** | RadioMaster TX16S Mk II (AG01 Gimbals) + ELRS 900MHz TX/RX | High-reliability, ultra-low latency command & control |
| **Telemetry & Comms** | Holybro SiK 500mW 915MHz + Long-Range LoRa + 4G/5G Cellular | Real-time drone telemetry & high-speed image payload transmission |
| **Propulsion** | 2x SunnySky X2814 Brushless Motors + 40A+ BLHeli ESCs | Dual-motor setup, ~2020g thrust per motor with APC 11x5.5 props |
| **Power System** | 4S Li-ion 8000mAh+ Main Battery Pack | High energy-density power for extended flight times in cold alpine air |
| **Payload** | High-Resolution Camera + Onboard SSD + Gimbal | Sub-centimeter 3D moraine modeling & photogrammetry |

---

## 6. Radar (Sentinel-1 SAR) Cloud-Penetration Pipeline

To overcome optical cloud blindness during South Asian monsoons (peak melt season):
* **Script:** `sentinel1_processing.py`
* **Technology:** Copernicus Sentinel-1 Synthetic Aperture Radar (C-band SAR, IW mode, 12-day revisit)
* **Processing Modes:**
  * **Mode A (Google Earth Engine):** Cloud computation, VV and VH polarization, VH/VV cross-ratio, SRTM DEM slope and elevation stacking, Random Forest 3-class segmentation.
  * **Mode B (Local Rasterio/GDAL):** Offline GeoTIFF processing and GeoJSON polygon extraction.
* **Impact:** Provides weather-independent, day/night continuity when optical sensors are obscured.

---

## 7. Repository Directory Structure & File Map

```
d:/NIDM-SRM Drone Project Docs/
├── Architecture_Document.md           # Formal Event-Driven & Relational ER architecture document
├── Final_Project_Report.md            # Comprehensive final project report (Dissertation format)
├── Review_2_Final_PPT.pptx            # 43-slide presentation prepared for Minor Project Review 2
├── Review_2_Mega_Document.md          # Full technical review documentation for Review 2
├── Review_2_PPT_Script.txt            # Slide-by-slide presentation speaking script and guidelines
├── generate_ppt.py                    # Python-pptx script automating the creation of Review 2 slides
├── main_updated.py                    # Core pipeline orchestration script (DEM analysis, indices, ML)
├── sentinel1_processing.py            # Complete Sentinel-1 SAR 3-class classifier (GEE & Local modes)
├── data_processing.py                 # DEM preprocessing and raster array transformations
├── dem_analysis.py                    # Topographic analysis (slope, aspect, curvature, hillshade)
├── indices_calculation.py             # Indices (TWI, SPI, CTI, TRI, VRM, flow accumulation)
├── ml_classification.py               # ML training models (Decision Tree, Random Forest, SVM)
├── calculate_ndvi.py                  # Standalone NDVI calculation utility
├── dump_tables.py / make_table_img.py # Table serialization and visual snapshot utilities
├── requirements.txt                   # Project Python dependencies (rasterio, geopandas, etc.)
│
├── Documents/                         # Official NIDM Grant & Institutional Records
│   ├── Acceptence Letter/             # Formal acceptance from SRMIST / NIDM
│   ├── Bank-Accounts and Fund Transfers/ # Grant financial tracking
│   ├── Inception Report-New/          # Inception Report V2 submitted to NIDM
│   ├── Presentations/                 # Grant defense & Review Meeting 1 presentations
│   ├── Sanction Order/                # Official NIDM Sanction Order (RP24, 8 Lakhs)
│   ├── Student Internship/            # Recruitment records for project interns
│   └── ToR/                           # Official NIDM Terms of Reference (ToR)
│
├── Drone Setup/                       # Tier 2 Fixed-Wing UAV Hardware
│   ├── Drone Project- RTF - NCO4G.xlsx # Full Bill of Materials (BOM) & hardware specs
│   └── Relevant Papers-Drone-Setup/   # Cryospheric drone photogrammetry research literature
│
├── GLOF project/                      # Exploratory notebooks & early analysis
│   ├── GLOF AI ML DL2.ipynb           # Comprehensive deep learning & ML exploratory notebook
│   └── TODO.MD                        # Initial 3-class classification goals
│
├── SPRINT2/                           # Sprint II Deliverables & Assets
│   ├── Architecture Document.pdf      # Sprint II architecture documentation
│   ├── FUNCTIONAL DOCUMENT.pdf        # Sprint II functional specifications
│   ├── FUNCTIONAL TEST CASES.pdf      # Test cases (TC-SP2-01 to 04)
│   ├── SPRINT RETROSPECTIVE.pdf       # Sprint retrospective
│   ├── Sprint2_Documents_Content.txt  # Detailed textual content for Sprint II
│   └── generate_review3_assets.py     # Asset rendering script
│
├── internship 2.0/                    # Multi-temporal classification scripts
│   ├── accurate_glacier_classifier.py # High-precision optical classification logic
│   ├── classify_glacier_3class.py     # 3-zone classifier (Glacier, Lake, Non-Glacier)
│   ├── detect_turquoise_lakes.py      # Spectral isolation for sediment-rich glacial lakes
│   └── generate_glacier_geojson.py    # Vector GeoJSON generation routines
│
├── intership/                         # Data calibration and diagnostics
│   ├── DIAGNOSIS_REPORT.md            # Resolution of synthetic vs. Sikkim real-coordinate data
│   ├── README_Image_Reading.md        # Guide on rasterio/GDAL satellite ingestion
│   └── fix_vrt_paths.py               # Repair utilities for Virtual Raster (VRT) tile paths
│
└── Data / Data_Aligned / Satellite Image Processing / final_glacier_maps / lake_change_results
                                       # Input imagery, aligned rasters, GeoJSON vectors, and results
```

---

## 8. Summary of Key Achievements to Date

1. **35-Year Multi-Sensor Temporal Pipeline:** Successfully ingested and normalized Landsat-5 (1988–2011) and Sentinel-2 (2021–2025, 66 scenes) to prove decadal retreat.
2. **Automated Watershed Delineation:** Extracted 413 micro-watersheds from ASTER GDEM using D8 hydrological flow modeling without manual intervention.
3. **Robust 3-Class ML Classifier:** Deployed a Decision Tree / Random Forest classifier with tri-spectral indices ($NDWI, NDSI, NDVI$) that eliminated shadow-induced false positives.
4. **Physical Morphometry Indexing:** Computed $AAR$, $SI$, $L/W$, and $CCI$ across all 413 watersheds, isolating 2 high-hazard priority outburst zones in Sikkim.
5. **Standardized Disaster Intelligence:** Serialized threat metrics into relational spreadsheets and GeoJSON vectors keyed by `W_ID`.
6. **Field Reconnaissance:** Validated theoretical model parameters against physical damage in North Sikkim from the 2023 Chungthang event.
7. **Monsoon Cloud Resilience:** Developed the Sentinel-1 SAR microwave radar classifier for all-weather monitoring.
8. **Drone System Design:** Finalized the hardware, avionics, and telemetry architecture for tactical fixed-wing UAV deployment.

---

## 9. Immediate Roadmap & Future Work

* [ ] **Physical Drone Assembly & Flight Testing:** Procure and integrate the specified airframe, Cube Orange FC, and ELRS/LoRa telemetry; execute high-altitude test flights.
* [ ] **End-to-End Cloud Deployment:** Transition python ingestion routines to automated serverless Cloud functions triggered by GEE satellite revisit events.
* [ ] **Interactive Web Dashboard:** Build a modern web interface (React / MapLibre / Leaflet) displaying the 413 micro-watersheds, hazard tiers, and time-lapse vector dynamics for civil defense operators.
* [ ] **Peer-Reviewed Publications:** Finalize journal manuscripts and conference papers documenting the multi-platform framework and Sikkim field validation.
