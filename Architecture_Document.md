# Architecture Document: GLOF Vulnerability Classification Pipeline

Based on the implemented methodology for the NIDM-Funded GLOF Study, below is the formal architectural breakdown tailored specifically to your pipeline's design, ready to submit for this assignment requirement.

---

## 1. Application Architecture

**Selected Architecture: Event-Driven Architecture (Data Pipeline)**

**Detailed Explanation:**
The GLOF monitoring system is constructed as an **Event-Driven Data Processing Pipeline**. Unlike standard user-facing applications, this system does not require continuous microservices. Instead, it relies on temporal "events"—specifically the scheduled overpass and data availability of Sentinel-2 (5-day revisit) or Sentinel-1 SAR (12-day revisit) satellites. 

When a new satellite scene is acquired from Google Earth Engine (GEE), this "data event" triggers an automated, sequential execution chain:
1.  **Ingestion Event:** The raw `.tif` arrays and ASTER GDEM are loaded into system memory.
2.  **Processing Event:** Calculation of Multi-Spectral Indices (NDWI/NDSI) triggers the Machine Learning Decision Tree (DT).
3.  **Analytics Event:** The DT outputs trigger the morphometric computation equations (AAR, SI, L/W Ratio, CCI) across the 413 micro-watersheds.
4.  **Output Event:** The processed data culminates in the generation of the 5-Tier GLOF Susceptibility Index.

By utilizing an event-driven flow, the pipeline operates efficiently, utilizing computational resources only when fresh remote-sensing matrices are injected into the ingestion folder.

*(Note for your assignment: Use the PlantUML architecture diagram we generated earlier in the chat as your "deployment/component diagram" for this section!)*

---

## 2. Database Architecture

Because this is a geospatial research application, it does not rely on a standard SQL server (like MySQL or Postgres). Instead, it utilizes a **Flat-File Relational Database Architecture** serialized into `.xlsx` and `.geojson` formats, linked universally by a unified primary key.

### 2.1 ER Diagram (Entity-Relationship Logic)
The entire analytical database is relationally bound by a single Primary Key: **`W_ID`** (Micro-Watershed ID: 1 through 413).
*   **Entity 1: Glacier** (`Glacier parameters.xlsx`) 
    *   *Attributes:* `W_ID` (PK), Accumulation Area, Snout Index, AAR.
*   **Entity 2: Glacial Lake** (`Glacial lake parameters.xlsx`) 
    *   *Attributes:* `W_ID` (PK), Lake Length, Mean Width, L/W Ratio.
*   **Entity 3: GLOF Warning System** (`GLOF Index.xlsx`)
    *   *Attributes:* `W_ID` (PK), CCI, Decision Tree Hazard Score (Very High to Very Low).
*   *Relationship:* One-to-One. Each distinct Micro-Watershed (`W_ID`) mathematically relates to one parent glacier and one terminal lake, culminating in exactly one final Warning Score.

### 2.2 Schema Design
**Table: `tb_glof_index` (The final serialized output)**
*   `W_ID` (Integer, Primary Key)
*   `ICI` (Float, Ice Compactness Index)
*   `SI` (Float, Snout Index)
*   `AAR` (Float, Accumulation Area Ratio)
*   `CCI` (Float, Catchment Compactness Index)
*   `Hazard_Score` (String/Categorical, Values: Very High, High, Medium, Low, Very Low)
*   `Year` (Date/Timestamp)

---

## 3. Data Exchange Contract

As an autonomous remote-sensing framework, the system frequently passes vast amounts of geospatial geometry between deep learning components, GIS software, and external web engines.

### 3.1 Frequency of Data Exchanges
*   **Dynamic / Temporal:** Exchanges occur every 5 to 12 days, rigidly aligning with the orbital revisit cycle of the Sentinel-2 (Optical) and Sentinel-1 (SAR) constellations over the Sikkim topographies. 
*   **Batch / Historical:** Exchanges occur via bulk requests when evaluating multi-decadal historic degradation using the Landsat legacy archives (1988–2023).

### 3.2 Data Sets
The pipeline interacts with three disparate data schemas:
1.  **Multi-Spectral Raster Matrices:** Raw and processed Top-of-Atmosphere spatial data preserving structural 10-meter metadata (`.tif` format).
2.  **Coordinate Vector Polygonization:** Sharp, lightweight geometric latitude/longitude boundaries of the lakes and glaciers (`.geojson` format).
3.  **Numerical Morphometry:** The finalized physical parameters utilized for threat assessment (`.xlsx` / `.csv` tabular formats).

### 3.3 Mode of Exchanges
*   **API (Application Programming Interface):** Data ingestion is handled via the Google Earth Engine (GEE) Python API, directly establishing a data-exchange contract with the Copernicus and USGS satellite servers to pull target temporal scenes.
*   **File I/O:** Internal processing outputs are exchanged dynamically through local memory rendering (via Pandas and Rasterio) to read and serialize heavy spatial data into lightweight `.csv` and `.geojson` architectures without overwhelming system RAM.
