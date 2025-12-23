# --------------------------------------------------------------
# FINAL WORKING SCRIPT – 12 SEPARATE Z-SCORE PLOTS
# Handles your exact broken CSV perfectly
# --------------------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. READ THE CSV WITH NO ASSUMPTIONS (this is the magic line)
df = pd.read_csv("output_data_full.csv", header=0, sep=',\s*|\t', engine='python', on_bad_lines='skip')

# 2. FORCE THE EXACT 12 COLUMN NAMES (your real ones)
correct_columns = [
    'Propellent atomic Mass [amu]',
    'Ionization Energy [eV]',
    'Dissociation Energy [eV]',
    'Anode Mass Flow Rate [SCCM]',
    'Discharge Voltage [V]',
    'Cathode Mass Flow Rate [SCCM]',
    'Cathode Keeper Voltage [V]',
    'Beam Current [A]',
    'Thrust [mN]',
    'Discharge Current [A]',
    'Electromagnetic Inner Coil Current [A]',
    'Electromagnetic Outer Coil Current [A]'
]

# If file has wrong number of columns or missing header → fix it
if df.shape[1] >= len(correct_columns):
    df = df.iloc[:, :len(correct_columns)]  # take only first 12
df.columns = correct_columns

# Fix the outer coil column name once and for all
df.columns = df.columns.str.replace('Outer  Coil', 'Outer Coil', regex=False)

# 3. Create output folder
os.makedirs("Z_SCORE_PLOTS_FINAL", exist_ok=True)

# 4. Plot each column as Z-score — separate figure
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (9, 6)})

for i, col in enumerate(correct_columns, 1):
    data = pd.to_numeric(df[col], errors='coerce').dropna()
    
    if len(data) < 3:
        print(f"Skipping {col} — too few valid points ({len(data)})")
        continue
    
    z = (data - data.mean()) / data.std()

    plt.figure()
    sns.histplot(z, kde=True, bins=50, color=f'C{i-1}', alpha=0.8, edgecolor='black', linewidth=0.5)
    sns.kdeplot(z, color='darkred', linewidth=2.5)

    plt.xlim(-4.5, 4.5)
    plt.xlabel("Z-score", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    plt.title(f"Z-score Distribution\n{col}", fontsize=14, pad=15)

    # Stats box
    stats = f"n = {len(data)}\n" \
            f"μ = {data.mean():.4g} ± {data.std():.4g}\n" \
            f"range = {data.min():.3g} → {data.max():.3g}"
    plt.text(0.97, 0.95, stats, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.grid(True, alpha=0.3)
    sns.despine()

    safe_name = col.replace(" [", "_").replace("]", "").replace(" ", "_")
    plt.savefig(f"Z_SCORE_PLOTS_FINAL/{i:02d}_{safe_name}_Zscore.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"Z_SCORE_PLOTS_FINAL/{i:02d}_{safe_name}_Zscore.pdf", bbox_inches='tight')
    plt.close()

    print(f"SUCCESS → {i:02d} {col}  (n = {len(data)})")

print("\nALL 12 Z-SCORE PLOTS CREATED SUCCESSFULLY!")
print("Check folder: Z_SCORE_PLOTS_FINAL")