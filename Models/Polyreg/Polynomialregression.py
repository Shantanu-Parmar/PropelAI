# ===============================================================
# POLYNOMIAL REGRESSION – THRUSTER THRUST PREDICTION (2025 FINAL)
# Now with Anode Mass Flow Rate in g/s (no leading zeros)
# No filtering, pure numbers, full logging + plots
# ===============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ------------------- 1. SETTINGS -------------------
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

DATA_FILE = "final.csv"  # ← CHANGE TO YOUR FILE
RESULTS_DIR = "poly_regression_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"{RESULTS_DIR}/training_log_{TIMESTAMP}.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

log("POLYNOMIAL REGRESSION TRAINING STARTED")
log(f"Timestamp: {TIMESTAMP}")

# ------------------- 2. LOAD + CONVERT TO g/s -------------------
df = pd.read_csv(DATA_FILE)

# Convert kg/s → g/s (multiply by 1000)
df['Anode Mass Flow Rate [mg/s]'] = df['Anode Mass Flow Rate [mg/s]']

# Your exact columns (updated with g/s)
feature_cols = [
    'Anode Mass Flow Rate [mg/s]',  # ← now in g/s
    'Beam Current [A]',
    'Cathode Keeper Voltage [V]',
    'Discharge Voltage [V]',
    'Discharge Current [A]',
    'Electromagnetic Inner Coil Current [A]',
    'Electromagnetic Outer Coil Current [A]'
]
target_col = 'Thrust [mN]'

X = df[feature_cols].values
y = df[target_col].values

log(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
log(f"Thrust range: {y.min():.3f} to {y.max():.3f} mN")
log(f"Anode Flow range: {df['Anode Mass Flow Rate [mg/s]'].min():.3f} to "
    f"{df['Anode Mass Flow Rate [mg/s]'].max():.3f} mg/s") 

# ------------------- 3. TRAIN/TEST SPLIT -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

log(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ------------------- 4. TRAIN DEGREES 1-4 -------------------
results = []

for degree in [1, 2, 3, 4]:
    log(f"\n--- Training Polynomial Degree {degree} ---")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly',   PolynomialFeatures(degree, include_bias=False)),
        ('model',  LinearRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2  = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    results.append({
        'degree': degree,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'model': pipeline
    })
    
    log(f"Degree {degree} to Test R²: {test_r2:.5f} | MAE: {test_mae:.3f} mN | RMSE: {test_rmse:.3f} mN")

# ------------------- 5. PICK BEST MODEL -------------------
best = max(results, key=lambda x: x['test_r2'])
log("\n" + "="*60)
log(f"BEST MODEL: Degree {best['degree']}")
log(f"Test R²  = {best['test_r2']:.6f}")
log(f"Test MAE = {best['test_mae']:.3f} mN")
log(f"Test RMSE = {best['test_rmse']:.3f} mN")

# ------------------- 6. FINAL PLOTS -------------------
y_pred = best['model'].predict(X_test)

# Scatter: actual vs predicted
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
plt.xlabel("Measured Thrust [mN]")
plt.ylabel("Predicted Thrust [mN]")
plt.title(f"Best Polynomial Model (Degree {best['degree']})\n"
          f"R^2 = {best['test_r2']:.5f} | MAE = {best['test_mae']:.3f} mN")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/best_model_scatter_{TIMESTAMP}.png", dpi=300)
plt.show()

# Residuals
residuals = y_test - y_pred
plt.figure()
sns.histplot(residuals, kde=True, bins=50)
plt.xlabel("Residual (Measured - Predicted) [mN]")
plt.title(f"Residual Distribution (MAE = {best['test_mae']:.3f} mN)")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/residuals_{TIMESTAMP}.png", dpi=300)
plt.show()

# ------------------- 7. SAVE EVERYTHING -------------------
joblib.dump(best['model'], f"{RESULTS_DIR}/BEST_polynomial_thruster_model.pkl")
joblib.dump(results, f"{RESULTS_DIR}/all_models_results.pkl")

# Save summary
summary = pd.DataFrame([{
    'degree': r['degree'],
    'train_r2': r['train_r2'],
    'test_r2': r['test_r2'],
    'test_mae_mN': r['test_mae'],
    'test_rmse_mN': r['test_rmse']
} for r in results])
summary.to_csv(f"{RESULTS_DIR}/summary_{TIMESTAMP}.csv", index=False)

log("\nALL DONE! Results saved in:")
log(RESULTS_DIR)
log("Best model: BEST_polynomial_thruster_model.pkl")
log("Use it like: model = joblib.load('...pkl'); pred = model.predict(new_data)")

print("\nTRAINING COMPLETE. CHECK FOLDER:", RESULTS_DIR)