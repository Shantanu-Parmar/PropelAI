import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
import os

# ------------------- 1. Load & fix columns -------------------
df = pd.read_csv("output_data_full.csv", sep=',\s*|\t', engine='python', on_bad_lines='skip', header=0)

# Force exact column names
df.columns = [
    'Propellent atomic Mass [amu]', 'Ionization Energy [eV]', 'Dissociation Energy [eV]',
    'Anode Mass Flow Rate [SCCM]', 'Discharge Voltage [V]', 'Cathode Mass Flow Rate [SCCM]',
    'Cathode Keeper Voltage [V]', 'Beam Current [A]', 'Thrust [mN]',
    'Discharge Current [A]', 'Electromagnetic Inner Coil Current [A]',
    'Electromagnetic Outer Coil Current [A]'
]

# Fix any spacing issue
df.columns = df.columns.str.replace('Outer  Coil', 'Outer Coil')

# ------------------- 2. DROP unwanted columns -------------------
drop_cols = [
    'Propellent atomic Mass [amu]',
    'Ionization Energy [eV]',
    'Dissociation Energy [eV]',
    'Cathode Mass Flow Rate [SCCM]'   
]
df = df.drop(columns=drop_cols, errors='ignore')

# ------------------- 3. Define normalization groups -------------------

# A. Nearly perfect normal distributions → StandardScaler (Z-score)
# Equation:  x_z = (x - μ) / σ
perfect_normal_cols = [
    'Thrust [mN]',
    'Cathode Keeper Voltage [V]',
    'Discharge Voltage [V]',
    'Discharge Current [A]'
]

scaler_z = StandardScaler()
df_z = pd.DataFrame(
    scaler_z.fit_transform(df[perfect_normal_cols]),
    columns=[c + "_z" for c in perfect_normal_cols]
)

# B. Continuous but with mild right skew / clustered values / outliers → RobustScaler
# Equation:  x_robust = (x - median) / IQR     (IQR = Q75 - Q25)
robust_cols = [
    'Anode Mass Flow Rate [SCCM]',
    'Beam Current [A]',
    'Electromagnetic Inner Coil Current [A]',
    'Electromagnetic Outer Coil Current [A]'   
]

scaler_robust = RobustScaler()
df_robust = pd.DataFrame(
    scaler_robust.fit_transform(df[robust_cols]),
    columns=[c + "_robust" for c in robust_cols]
)

# ------------------- 4. Combine everything -------------------
df_normalized = pd.concat([
    df_z.reset_index(drop=True),
    df_robust.reset_index(drop=True)
], axis=1)

# ------------------- 5. Optional: physics-derived targets -------------------
raw = df.reset_index(drop=True)
df_normalized['Specific_Thrust_mN_per_SCCM'] = raw['Thrust [mN]'] / raw['Anode Mass Flow Rate [SCCM]']
df_normalized['Beam_to_Anode_Ratio'] = raw['Beam Current [A]'] / raw['Anode Mass Flow Rate [SCCM]'].replace(0, np.nan)

# Save
os.makedirs("normalized_final", exist_ok=True)
df_normalized.to_csv("normalized_final/thruster_normalized_FINAL.csv", index=False)

print("FINAL NORMALIZATION COMPLETE!")
print(f"Shape: {df_normalized.shape}")
print(f"Columns ({df_normalized.shape[1]} total):")
for c in df_normalized.columns:
    print(f"  → {c}")