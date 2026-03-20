# Revealing Human Mobility Patterns to Extreme Heat through Street Network Topology

This repository contains the analysis code for a study examining how extreme heat events shape human mobility patterns across 20 major US Metropolitan Statistical Areas (MSAs). Using large-scale mobility data linked with temperature records, NAICS-coded POIs' visits, and OpenStreetMap street network data, the study identifies divergent behavioral responses to heat stress and connects them to urban form characteristics.

## Research Overview

In this study, visitation frequency is defined as the number of visits to POIs within the home city, originating from home, during a specific time period. We extend the state-of-the-art by addressing the following questions:

- How do the spatiotemporal dynamics of human mobility under extreme heat vary across major US metropolitan areas?
- What are the revealed visitation patterns to urban establishments and essential services under extreme heat?
- How do the topological characteristics of street networks, as representations of urban morphology, shape human mobility patterns under extreme heat?

Cities are classified into three behavioral clusters derived from Dynamic Time Warping (DTW) hierarchical clustering:

Heat-Sensitive MSAs: Chicago, Dallas, Houston

Temperature-Resilient MSAs: Los Angeles, New York, Phoenix

Adaptive / Baseline MSAs: Austin, Charlotte, Columbus, Denver, Indianapolis, Jacksonville, Nashville, Oklahoma City, Philadelphia, San Antonio, San Diego, San Francisco, San Jose, Seattle

**Research framework**
<img width="975" height="1104" alt="image" src="https://github.com/user-attachments/assets/abd11693-eab1-4817-9a86-0664387d007c" />

---

## Repository Structure

```
Code/
├── Data/                                   # Input data directory (see Data section)
├── F1. MSAs_Temperature_Panel.py           # Fig 1 — Temperature–essential visit panel (20 MSAs)
├── F2a. DTW_Hierarchy.py                   # Fig 2a — DTW hierarchical clustering & MDS
├── F2b.  Temp_NormVisit.py                 # Fig 2b — Temperature-normalized visit trends
├── F2c-e. Statistic_Temp_Dist_Vist.py      # Fig 2c–e — Violin plots (temp, distance, visitation)
├── F3a. Visit_Ration_Trend.py              # Fig 3a — Essential-service visit ratio vs temperature
├── F3c. Walkable POIs.py                   # Fig 3c — Walkable POI ratio: heat vs non-heat days
├── F3d. City_Weather_NAICS_Analysis.py     # Fig 3d — Visit change by NAICS category & cluster
├── F4a_T2. Street_Network_Orentation.py    # Fig 4a/Table 2 — Street orientation polar plots & metrics
├── F4b. Classifed_Network_Orentation.py    # Fig 4b — Classified orientation by city cluster
├── T1. Spearman Test.py                    # Table 1 — Spearman correlation tests between street network metrics and mobility indicators
└── Supplementary/                          # Supplementary figures and tables
```

---

## Supplementary Information

The `Supplementary/` directory contains code for all figures and tables in the Supplementary Information of the paper. Each script is prefixed with its corresponding item number (e.g., `S3. Silhouette_Analysis.py` is for Fig. S3, `T1. Spearman Test.py` is for Table 1).

## Data

The study covers 23,460 CBGs, with a total of approximately 163 million person-trips and a total travel distance of about 1.31×108 km. Raw data are not included in this repository due to size and licensing constraints. Please refer to the data availability statement in the associated publication. Scripts expect input files to be placed under a `Data/` directory. The required folder and file names correspond directly to the paths defined at the top of each script (e.g., `DATA_DIR = r"path/to/Data/CBG_City_Week_Statistics"`).
Mobility data are derived from anonymized, aggregated mobile device records; temperature data are from NOAA daily summaries; street network data are retrieved at runtime from OpenStreetMap via OSMnx.

---

## Requirements

Python 3.9+ is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

**Core dependencies:**

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
tqdm
osmnx
networkx
openpyxl
```

---

## Usage

Each script is self-contained and can be run independently. Before running, update the hardcoded paths at the top of each file to point to your local `Data/` directory. For example, in `F2a. DTW_Hierarchy.py`:

```python
DATA_DIR   = r"path/to/Data/CBG_City_Week_Statistics"
OUTPUT_DIR = r"path/to/output"
```

Then run:

```bash
python "F1. MSAs_Temperature_Panel.py"
python "F2a. DTW_Hierarchy.py"
# ... and so on


## Contact

For questions regarding the code or data, please open an issue in this repository or contact the corresponding author at kukujoyyo@outlook.com.
