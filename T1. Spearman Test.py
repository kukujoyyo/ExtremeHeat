import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
import os
warnings.filterwarnings('ignore')

BASE_PATH  = r'D:\Research\ExtremeHeat\Preprocess\Journal\Code\Data'
INPUT_FILE = os.path.join(BASE_PATH, 'Merge_Street_Visitor.xlsx')
OUTPUT_PATH = r'D:\Research\ExtremeHeat\Preprocess\Journal\Code'
os.makedirs(OUTPUT_PATH, exist_ok=True)

LEGACY_MAP = {
    'High Temperature Suppressed': 'Heat-sensitive MSAs',
    'Mid Temperature Stable':      'Temperature-resilient MSAs',
    'Large Metro Baseline':        'Adaptive/baseline MSAs',
}

df = pd.read_excel(INPUT_FILE)
df['Category'] = df['Category'].replace(LEGACY_MAP)
df_clean = df.dropna(subset=['Visit_Change_Rate', 'Distance_Change_Rate'])

dependent_vars   = ['Visit_Change_Rate', 'Distance_Change_Rate']
independent_vars = ['Orientation_Entropy', 'Order_Phi', 'Node_Density', 'Clustering_Coeff', 'BC']
categories       = df_clean['Category'].unique()

results = []

for category in categories:
    subset = df_clean[df_clean['Category'] == category]
    for dep_var in dependent_vars:
        for ind_var in independent_vars:
            s_corr, s_pval = spearmanr(subset[ind_var], subset[dep_var])
            s_sig = '***' if s_pval < 0.001 else ('**' if s_pval < 0.01 else ('*' if s_pval < 0.05 else 'ns'))
            results.append({
                'Category':             category,
                'Dependent_Variable':   dep_var,
                'Independent_Variable': ind_var,
                'Sample_Size':          len(subset),
                'Spearman_rho':         s_corr,
                'Spearman_p':           s_pval,
                'Spearman_Sig':         s_sig,
            })

output_file = os.path.join(OUTPUT_PATH, 'Table 1. Spearman Test.xlsx')
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    pd.DataFrame(results).to_excel(writer, sheet_name='Spearman test', index=False)

print(f"✓ Saved: {output_file}")
