import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ====================== CHANGE THESE TWO LINES ======================
csv_file = "output_data_full.csv"       # your CSV file
output_folder = "Hall_Thruster_Plots"   # folder where images will be saved
# ====================================================================

os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()

# Force numeric (non-numeric → NaN)
for col in df.columns:
    if col not in ['Propellant', 'Date', 'Comment', 'Notes']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ====================== Derived quantities ======================
g0 = 9.80665
mass_col = [c for c in df.columns if 'atomic Mass' in c or 'amu' in c.lower()][0]
print(f"Using atomic mass from column: {mass_col}")
df['Anode mass flow [kg/s]'] = df['Anode Mass Flow Rate [SCCM]'] * 9.73e-8 * (df[mass_col] / 131.293)

df['Power [W]']               = df['Discharge Voltage [V]'] * df['Discharge Current [A]']
df['Thrust [N]']              = df['Thrust [mN]'] * 1e-3
df['Isp [s]']                 = df['Thrust [N]'] / (df['Anode mass flow [kg/s]'] * g0)
df['Total Efficiency [%]']    = 100 * (df['Thrust [N]']**2) / (2 * df['Anode mass flow [kg/s]'] * df['Power [W]'])
df['Anode Efficiency [%]']    = 100 * (df['Thrust [N]']**2) / (2 * df['Anode mass flow [kg/s]'] * df['Power [W]'])
df['Current Utilization [%]'] = 100 * df['Beam Current [A]'] / df['Discharge Current [A]']

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

# ====================== Plotting function ======================
def save_fig(name):
    plt.savefig(f"{output_folder}/{name}.png", dpi=300, bbox_inches='tight')
    plt.close()

# 1. Performance Map 
plt.figure(figsize=(8,6))
sc = plt.scatter(df['Isp [s]'], df['Thrust [mN]'], c=df['Power [W]'], cmap='viridis', s=100, edgecolors='k', linewidth=0.5)
plt.colorbar(sc, label='Power [W]')
plt.xlabel('Specific Impulse [s]')
plt.ylabel('Thrust [mN]')
plt.title('Performance Map – Thrust vs Isp')
save_fig("01_Performance_Map")

# 2. Thrust vs Power
plt.figure(figsize=(8,6))
sc = plt.scatter(df['Power [W]'], df['Thrust [mN]'], c=df['Anode Mass Flow Rate [SCCM]'], cmap='plasma', s=100, edgecolors='k')
plt.colorbar(sc, label='Anode flow [SCCM]')
plt.xlabel('Discharge Power [W]')
plt.ylabel('Thrust [mN]')
plt.title('Thrust vs Discharge Power')
save_fig("02_Thrust_vs_Power")

# 3. Total Efficiency vs Power
plt.figure(figsize=(8,6))
sc = plt.scatter(df['Power [W]'], df['Total Efficiency [%]'], c=df['Discharge Voltage [V]'], cmap='magma', s=100)
plt.colorbar(sc, label='Discharge Voltage [V]')
plt.xlabel('Power [W]')
plt.ylabel('Total Efficiency [%]')
plt.title('Total Efficiency vs Power')
save_fig("03_Total_Efficiency_vs_Power")

# 4. Isp vs Discharge Voltage
plt.figure(figsize=(8,6))
sc = plt.scatter(df['Discharge Voltage [V]'], df['Isp [s]'], c=df['Anode Mass Flow Rate [SCCM]'], cmap='cividis', s=100)
plt.colorbar(sc, label='Anode flow [SCCM]')
plt.xlabel('Discharge Voltage [V]')
plt.ylabel('Isp [s]')
plt.title('Specific Impulse vs Discharge Voltage')
save_fig("04_Isp_vs_Voltage")

# 5. Current Utilization
plt.figure(figsize=(8,6))
plt.scatter(df['Discharge Current [A]'], df['Beam Current [A]'], c=df['Power [W]'], cmap='turbo', s=80)
mx = df['Discharge Current [A]'].max() * 1.05
plt.plot([0, mx], [0, mx], 'r--', lw=2, label='100 % utilization')
plt.xlabel('Discharge Current [A]')
plt.ylabel('Beam Current [A]')
plt.legend()
plt.title('Current Utilization')
save_fig("05_Current_Utilization")

# 6–8. Magnetic field sensitivity (if you have coil currents)
if 'Electromagnetic Inner Coil Current [A]' in df.columns:
    plt.figure(figsize=(8,6))
    plt.scatter(df['Electromagnetic Inner Coil Current [A]'], df['Thrust [mN]'], c=df['Power [W]'], cmap='coolwarm', s=100)
    plt.colorbar(label='Power [W]')
    plt.xlabel('Inner Coil Current [A]')
    plt.ylabel('Thrust [mN]')
    plt.title('Thrust vs Inner Coil Current')
    save_fig("06_Thrust_vs_Inner_Coil")

if 'Electromagnetic Outer' in ' '.join(df.columns):
    # assuming column contains "Outer" somewhere
    outer_col = [c for c in df.columns if 'outer' in c.lower()][0]
    plt.figure(figsize=(8,6))
    plt.scatter(df[outer_col], df['Thrust [mN]'], c=df['Power [W]'], cmap='hot', s=100)
    plt.colorbar(label='Power [W]')
    plt.xlabel(outer_col)
    plt.ylabel('Thrust [mN]')
    plt.title('Thrust vs Outer Coil Current')
    save_fig("07_Thrust_vs_Outer_Coil")

# 8. Efficiency heatmap in operating space
plt.figure(figsize=(8,7))
sc = plt.scatter(df['Anode Mass Flow Rate [SCCM]'], df['Discharge Voltage [V]'],
                 c=df['Total Efficiency [%]'], cmap='RdYlGn', s=120, edgecolors='k', linewidth=0.5)
plt.colorbar(sc, label='Total Efficiency [%]')
plt.xlabel('Anode Mass Flow Rate [SCCM]')
plt.ylabel('Discharge Voltage [V]')
plt.title('Total Efficiency in (flow × voltage) Space')
save_fig("08_Efficiency_Operating_Space")

# Interactive Plotly version (saved as HTML)
import plotly.express as px
fig = px.scatter(df, x='Isp [s]', y='Thrust [mN]', color='Power [W]',
                 size='Anode Mass Flow Rate [SCCM]', hover_data=df.columns,
                 title='Interactive Performance Map – Thrust vs Isp')
fig.write_html(f"{output_folder}/Interactive_Performance_Map.html")

print(f"\nAll done! {len([f for f in os.listdir(output_folder) if f.endswith('.png')])} PNG files + 1 HTML saved in folder: {output_folder}")