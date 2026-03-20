# Extreme Heat and Urban Mobility: Evidence from 20 U.S. Metropolitan Areas

This repository contains the analysis code for a study examining how extreme heat events shape human mobility patterns across 20 major U.S. Metropolitan Statistical Areas (MSAs). Using large-scale mobility data linked with temperature records, NAICS-coded point-of-interest visits, and OpenStreetMap street network data, the study identifies divergent behavioral responses to heat stress and connects them to urban form characteristics.

## Research Overview

Extreme heat is one of the deadliest climate hazards, yet its impacts on everyday urban mobility remain poorly understood at the metropolitan scale. This study asks:

- How does rising temperature reshape visits to essential services (healthcare, grocery stores, restaurants)?
- Do cities respond to heat stress uniformly, or are there distinct behavioral clusters?
- What urban-form features — walkable POI access, street network geometry — correlate with heat resilience?

Cities are classified into three behavioral clusters derived from Dynamic Time Warping (DTW) hierarchical clustering:

| Cluster | Cities |
| **Heat-Sensitive MSAs** | Chicago, Dallas, Houston |
| **Temperature-Resilient MSAs** | Los Angeles, New York, Phoenix |
| **Adaptive / Baseline MSAs** | Austin, Charlotte, Columbus, Denver, Indianapolis, Jacksonville, Nashville, Oklahoma City, Philadelphia, San Antonio, San Diego, San Francisco, San Jose, Seattle |

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

The `Supplementary/` directory contains code for all figures and tables in the Supplementary Information of the paper. Scripts are prefixed with their corresponding supplementary item number.

| Script | Supplementary Item | Description |
| `S2. Temperature_response.py` | Fig. S2 | Per-city temperature–visit response curves with smoothed trend lines |
| `S3. Silhouette_Analysis.py` | Fig. S3 | Silhouette score analysis for selecting the optimal number of DTW clusters |
| `S4. Proportional Distance_Visit frequency.py` | Fig. S4 | Proportional distance and visit frequency by NAICS category across city clusters |
| `S5. Distance_visit counts_temperature.py` | Fig. S5 | Boxplots of daily distance and visitor counts across temperature bins |
| `S9. Normality Tests.py` | Table S9 | Shapiro–Wilk normality tests for street network indicators |
| `S10a. Polar distribution of distance.py` | Fig. S10a | Polar histograms of travel distance direction by city |
| `S10b Polar distribution of visit.py` | Fig. S10b | Polar histograms of visit-pattern direction by city |
| `S11. Probability Density.py` | Fig. S11 | Probability density plots of mobility indicators |
| `S12. BC_Map.py` | Fig. S12 | Betweenness centrality maps for each city derived from OSMnx street networks |
| `S13. Cities_SNHMI_Map.py` | Fig. S13 | City-level Street Network Heat Mobility Index (SNHMI) map with Census and OSMnx data |
---

## Data

Raw data are not ncludedd in this repository due to size and licensing constraints. Please refer to the data availability statement in the associated publication. Scripts expect input files to be placed under a `Data/` directory. The required folder and file names correspond directly to the paths defined at the top of each script (e.g., `DATA_DIR = r"path/to/Data/CBG_City_Week_Statistics"`).
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
