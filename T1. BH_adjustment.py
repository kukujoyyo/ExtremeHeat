import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_ROOT  # noqa: E402

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control


# ==================== 参数 ====================

INPUT_FILE = Path(
    os.path.join(DATA_ROOT, "Results2_essential/Street_Heat/Stratified_Analysis_Complete_Results.xlsx")
)

OUTPUT_FILE = INPUT_FILE.with_name("Table1_Spearman_BH_Results.xlsx")

SHEET_NAME = "1_Stratified_Correlation"
ALPHA = 0.05
NUMBER_OF_TESTS = 30


# ==================== 显著性标记 ====================

def significance_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ==================== 读取Spearman结果 ====================

# 原表A:G：
# 类别、结果变量、指标、样本量、Spearman rho、原始星号、原始p值
df = pd.read_excel(
    INPUT_FILE,
    sheet_name=SHEET_NAME,
    usecols="A:G",
    nrows=NUMBER_OF_TESTS,
    engine="openpyxl",
)

df.columns = [
    "Category",
    "Outcome",
    "Indicator",
    "N",
    "Spearman_rho",
    "Original_stars",
    "Spearman_p",
]

# 原表类别和结果变量存在合并单元格，向下填充标签
df[["Category", "Outcome"]] = (
    df[["Category", "Outcome"]].ffill()
)

df["Spearman_p"] = pd.to_numeric(
    df["Spearman_p"], errors="raise"
)
df["Spearman_rho"] = pd.to_numeric(
    df["Spearman_rho"], errors="raise"
)

p_values = df["Spearman_p"].to_numpy(dtype=float)

# 校验原始p值
if len(p_values) != NUMBER_OF_TESTS:
    raise ValueError(
        f"应有{NUMBER_OF_TESTS}项检验，实际读取到{len(p_values)}项。"
    )

if (
    not np.isfinite(p_values).all()
    or ((p_values < 0) | (p_values > 1)).any()
):
    raise ValueError("原始p值必须为[0, 1]范围内的有效数值。")


# ==================== BH校正 ====================

# 对全部30项Spearman检验一起校正，不按类别分别校正
df["BH_adjusted_p"] = false_discovery_control(
    p_values,
    method="bh",
)

# 保留原始检验和BH校正后的显著性
df["Spearman_significant"] = df["Spearman_p"] < ALPHA
df["Spearman_stars"] = df["Spearman_p"].map(significance_stars)

df["BH_significant"] = df["BH_adjusted_p"] < ALPHA
df["BH_stars"] = df["BH_adjusted_p"].map(significance_stars)


# ==================== 完整结果 ====================

result = df[[
    "Category",
    "Outcome",
    "Indicator",
    "N",
    "Spearman_rho",
    "Spearman_p",
    "Spearman_stars",
    "Spearman_significant",
    "BH_adjusted_p",
    "BH_stars",
    "BH_significant",
]].copy()


# ==================== 分类及总体汇总 ====================

summary = (
    result.groupby("Category", sort=False)
    .agg(
        Number_of_tests=("Spearman_p", "size"),
        Spearman_significant_count=("Spearman_significant", "sum"),
        BH_significant_count=("BH_significant", "sum"),
    )
    .reset_index()
)

total = pd.DataFrame([{
    "Category": "Total",
    "Number_of_tests": len(result),
    "Spearman_significant_count": int(
        result["Spearman_significant"].sum()
    ),
    "BH_significant_count": int(
        result["BH_significant"].sum()
    ),
}])

summary = pd.concat([summary, total], ignore_index=True)

# 找出原始显著、BH校正后不再显著的关联
lost_significance = result.loc[
    result["Spearman_significant"] & ~result["BH_significant"]
].copy()


# ==================== 保存Excel ====================

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    result.to_excel(
        writer,
        sheet_name="Spearman_and_BH",
        index=False,
    )
    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False,
    )
    lost_significance.to_excel(
        writer,
        sheet_name="Lost_significance",
        index=False,
    )

    # 设置冻结表头、筛选和列宽
    for worksheet in writer.sheets.values():
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for column in worksheet.columns:
            width = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in column
            )
            worksheet.column_dimensions[
                column[0].column_letter
            ].width = min(max(width + 2, 12), 35)


# ==================== 显示结果 ====================

print("Spearman检验及BH校正完整结果：")
print(
    result.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print("\n显著关联数量汇总：")
print(summary.to_string(index=False))

print("\nBH校正后不再显著的关联：")
if lost_significance.empty:
    print("无")
else:
    print(
        lost_significance.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

print(f"\n结果文件：{OUTPUT_FILE}")
