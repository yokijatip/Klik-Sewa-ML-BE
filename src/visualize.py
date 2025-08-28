# src/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import numpy as np
import os

def load_test_data():
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    model = joblib.load('models/best_model.pkl')
    y_pred = model.predict(X_test)
    return y_test, y_pred

def plot_actual_vs_predicted():
    y_test, y_pred = load_test_data()
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Harga Aktual')
    plt.ylabel('Harga Prediksi')
    plt.title('Actual vs Predicted Price')
    plt.tight_layout()
    plt.savefig('reports/actual_vs_predicted.png')
    plt.show()

def plot_feature_importance():
    model = joblib.load('models/best_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    
    # Ambil feature names
    cat_features = preprocessor.transformers_[0][1].get_feature_names_out(['category', 'subcategory', 'city', 'district', 'condition', 'type'])
    text_features = preprocessor.transformers_[1][1].get_feature_names_out()
    feature_names = np.concatenate([cat_features, text_features])
    
    # Plot
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(10, 6))
    plt.title("Top 15 Fitur Paling Berpengaruh")
    plt.bar(range(15), importances[indices], color="skyblue", align="center")
    plt.xticks(range(15), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png')
    plt.show()

def plot_price_distribution():
    df = pd.read_csv('data/raw/dataset-sewa-360.csv')
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='category', y='daily_price')
    plt.title('Distribusi Harga Sewa Harian per Kategori')
    plt.ylabel('Harga Harian (Rp)')
    plt.tight_layout()
    plt.savefig('reports/price_by_category.png')
    plt.show()

if __name__ == "__main__":
    os.makedirs('reports', exist_ok=True)
    print("INFO: Mulai visualisasi...")
    plot_price_distribution()
    plot_feature_importance()
    plot_actual_vs_predicted()
    print("INFO: Visualisasi selesai! Lihat folder reports/")