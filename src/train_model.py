# src/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os
from datetime import datetime

# Load data
X = pd.read_csv('data/processed/X_processed.csv')
y = pd.read_csv('data/processed/y_daily.csv').values.ravel()

print(f"INFO: Dataset loaded - {X.shape[0]} samples, {X.shape[1]} features")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simpan X_test dan y_test untuk visualisasi nanti
os.makedirs('data/processed', exist_ok=True)
pd.DataFrame(X_test).to_csv('data/processed/X_test.csv', index=False)
pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)

print(f"INFO: Train set: {X_train.shape[0]} samples")
print(f"INFO: Test set: {X_test.shape[0]} samples")

# Inisialisasi model dengan parameter yang lebih baik
rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

gb = GradientBoostingRegressor(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Training
print("\nINFO: Melatih Random Forest...")
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("INFO: Melatih Gradient Boosting...")
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

# Evaluasi
def evaluate_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Hitung MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Hitung akurasi berdasarkan threshold ±20%
    within_20_percent = np.sum(np.abs((y_true - y_pred) / y_true) <= 0.20) / len(y_true) * 100
    
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'mape': float(mape),
        'accuracy_20_percent': float(within_20_percent),
        'mean_actual': float(np.mean(y_true)),
        'mean_predicted': float(np.mean(y_pred)),
        'std_actual': float(np.std(y_true)),
        'std_predicted': float(np.std(y_pred))
    }
    
    print(f"\nINFO: {name} Results:")
    print(f"  MAE  : Rp {mae:,.0f}")
    print(f"  RMSE : Rp {rmse:,.0f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  Accuracy (±20%): {within_20_percent:.1f}%")
    
    return metrics

# Evaluasi kedua model
rf_metrics = evaluate_model(y_test, rf_pred, "Random Forest")
gb_metrics = evaluate_model(y_test, gb_pred, "Gradient Boosting")

# Pilih model terbaik berdasarkan R²
if rf_metrics['r2_score'] >= gb_metrics['r2_score']:
    best_model = rf
    best_metrics = rf_metrics
    model_name = "RandomForest"
    print(f"\nINFO: Random Forest dipilih sebagai model terbaik (R² = {rf_metrics['r2_score']:.4f})")
else:
    best_model = gb
    best_metrics = gb_metrics
    model_name = "GradientBoosting"
    print(f"\nINFO: Gradient Boosting dipilih sebagai model terbaik (R² = {gb_metrics['r2_score']:.4f})")

# Siapkan metrics untuk disimpan
final_metrics = {
    "model_name": model_name,
    "training_date": datetime.now().isoformat(),
    "training_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "feature_count": int(X.shape[1]),
    "best_model_metrics": best_metrics,
    "random_forest_metrics": rf_metrics,
    "gradient_boosting_metrics": gb_metrics,
    
    # Metrics untuk API (format yang mudah diakses)
    "r2_score": best_metrics['r2_score'],
    "mae": best_metrics['mae'],
    "rmse": best_metrics['rmse'],
    "mape": best_metrics['mape'],
    "accuracy": best_metrics['accuracy_20_percent'],
    "confidence": min(best_metrics['r2_score'] * 100, 95),  # Confidence score (max 95%)
    
    # Dataset statistics
    "price_statistics": {
        "min_price": float(y.min()),
        "max_price": float(y.max()),
        "mean_price": float(y.mean()),
        "median_price": float(np.median(y)),
        "std_price": float(y.std())
    }
}

# Buat folder jika belum ada
os.makedirs('models', exist_ok=True)

# Simpan semua model dan metrics
joblib.dump(rf, 'models/random_forest.pkl')
joblib.dump(gb, 'models/gradient_boosting.pkl')
joblib.dump(best_model, 'models/best_model.pkl')

# Simpan predictions untuk analisis lebih lanjut
joblib.dump(rf_pred, 'models/rf_predictions.pkl')
joblib.dump(gb_pred, 'models/gb_predictions.pkl')

# Simpan metrics sebagai JSON
with open('models/model_metrics.json', 'w') as f:
    json.dump(final_metrics, f, indent=2)

print(f"\nINFO: [SUCCESS] Training selesai!")
print(f"INFO: Model terbaik: {model_name}")
print(f"INFO: R² Score: {best_metrics['r2_score']:.4f}")
print(f"INFO: MAE: Rp {best_metrics['mae']:,.0f}")
print(f"INFO: Akurasi (±20%): {best_metrics['accuracy_20_percent']:.1f}%")
print(f"INFO: Model dan metrics disimpan di folder 'models/'")

# Feature importance analysis (jika model support)
if hasattr(best_model, 'feature_importances_'):
    print("\nINFO: Menyimpan feature importance...")
    feature_importance = {
        'importances': best_model.feature_importances_.tolist(),
        'top_10_indices': np.argsort(best_model.feature_importances_)[-10:].tolist()
    }
    
    with open('models/feature_importance.json', 'w') as f:
        json.dump(feature_importance, f, indent=2)
    
    print("INFO: Feature importance disimpan di models/feature_importance.json")