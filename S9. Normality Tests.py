"""
Supplementary Table 4 — Descriptive statistics and normality tests
for street network indicators.

Input : D:\Research\ExtremeHeat\Preprocess\Results2_essential\Street_Heat\Merge_Street_Visitor_R.xlsx
Output: printed table + optional Excel export
"""

import pandas as pd
from scipy import stats

# ── 1. Load data ──────────────────────────────────────────────────────────────
FILE_PATH = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data\Merge_Street_Visitor.xlsx"

df = pd.read_excel(FILE_PATH, sheet_name="Detailed_Data")

# ── 2. Define variable groups ─────────────────────────────────────────────────
independent_vars = [
    "Orientation_Entropy",
    "Order_Phi",
    "Node_Density",
    "Clustering_Coeff",
    "BC",
]

dependent_vars = [
    "Visit_Change_Rate",
    "Distance_Change_Rate",
]

all_vars = independent_vars + dependent_vars

# Display names matching the paper
var_labels = {
    "Orientation_Entropy": "Orientation_Entropy (Hₒ)",
    "Order_Phi":           "Order_Phi (φ)",
    "Node_Density":        "Node_Density (N)",
    "Clustering_Coeff":    "Clustering_Coeff (C)",
    "BC":                  "BC",
    "Visit_Change_Rate":   "Visit_Change_Rate",
    "Distance_Change_Rate":"Distance_Change_Rate",
}

# ── 3. Compute statistics ─────────────────────────────────────────────────────
rows = []
for var in all_vars:
    col = df[var].dropna()

    mean     = col.mean()
    median   = col.median()
    skewness = col.skew()
    kurtosis = col.kurtosis()          # excess kurtosis (Fisher)

    stat, p = stats.shapiro(col)       # Shapiro-Wilk
    shapiro_w = stat

    # Significance stars
    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = ""

    rows.append({
        "Variables":    var_labels[var],
        "Mean":         round(mean,     3),
        "Median":       round(median,   3),
        "Skewness":     round(skewness, 3),
        "Kurtosis":     round(kurtosis, 3),
        "Shapiro-Wilk": round(shapiro_w,3),
        "Sig":          sig,
        "_group":       "Independent Variables" if var in independent_vars else "Dependent Variables",
        "_p":           p,
    })

result_df = pd.DataFrame(rows)

# ── 4. Print table ────────────────────────────────────────────────────────────
fmt_cols = ["Variables", "Mean", "Median", "Skewness", "Kurtosis", "Shapiro-Wilk", "Sig"]

print("\nSupplementary Table 4  Descriptive statistics and normality tests for street network indicators.\n")
print(f"{'Variables':<30} {'Mean':>8} {'Median':>8} {'Skewness':>10} {'Kurtosis':>10} {'Shapiro-Wilk':>13} {'Sig':>5}")
print("-" * 90)

current_group = None
for _, row in result_df.iterrows():
    if row["_group"] != current_group:
        current_group = row["_group"]
        print(f"\n  {current_group}")
    print(f"  {row['Variables']:<28} {row['Mean']:>8} {row['Median']:>8} "
          f"{row['Skewness']:>10} {row['Kurtosis']:>10} {row['Shapiro-Wilk']:>13}  {row['Sig']:>4}")

print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")

# ── 5. Export to Excel (optional) ─────────────────────────────────────────────
OUT_PATH = r"D:\Research\ExtremeHeat\Preprocess\Journal\Code\S9. Normality Tests.xlsx"

export_df = result_df[fmt_cols].copy()
export_df.to_excel(OUT_PATH, index=False)
print(f"\nResults saved to: {OUT_PATH}")
