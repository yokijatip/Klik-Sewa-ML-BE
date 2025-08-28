# src/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Load data
X = pd.read_csv('data/processed/X_processed.csv')
y = pd.read_csv('data/processed/y_daily.csv').values.ravel()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simpan X_test dan y_test untuk visualisasi nanti
os.makedirs('data/processed', exist_ok=True)
pd.DataFrame(X_test).to_csv('data/processed/X_test.csv', index=False)
pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)

# Inisialisasi model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Training
print("INFO: Melatih Random Forest...")
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("INFO: Melatih Gradient Boosting...")
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

# Evaluasi
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\nINFO: {name}")
    print(f"MAE  : Rp {mae:,.0f}")
    print(f"RMSE : Rp {rmse:,.0f}")
    print(f"R²   : {r2:.3f}")
    return r2

r2_rf = evaluate(y_test, rf_pred, "Random Forest")
r2_gb = evaluate(y_test, gb_pred, "Gradient Boosting")

# Pilih model terbaik
if r2_rf >= r2_gb:
    best_model = rf
    model_name = "RandomForest"
else:
    best_model = gb
    model_name = "GradientBoosting"

# Simpan semua model
joblib.dump(rf, 'models/random_forest.pkl')
joblib.dump(gb, 'models/gradient_boosting.pkl')
joblib.dump(best_model, 'models/best_model.pkl')

print(f"\nINFO: Model terbaik: {model_name} disimpan di models/best_model.pkl")