# src/preprocessing.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    
    # Debug: tampilkan nama kolom
    print("INFO: Kolom asli dari CSV:")
    for i, col in enumerate(df.columns):
        print(f"  [{i}] '{col}'")
    
    # Hapus spasi di nama kolom
    df.columns = df.columns.str.strip()
    
    # Cek apakah kolom terakhir bernama 'unit' padahal seharusnya 'type'
    if 'type' not in df.columns:
        if 'unit' in df.columns:
            print("INFO: Kolom 'unit' ditemukan. Mengganti nama menjadi 'type'...")
            df = df.rename(columns={'unit': 'type'})
        else:
            # Cari kolom yang isinya mirip tipe sewa
            possible_cols = [col for col in df.columns if 'type' in col.lower() or 'tipe' in col.lower()]
            if possible_cols:
                df = df.rename(columns={possible_cols[0]: 'type'})
                print(f"INFO: Kolom '{possible_cols[0]}' diubah menjadi 'type'")
            else:
                raise KeyError("Kolom 'type' tidak ditemukan dan tidak ada pengganti.")
    
    print("INFO: Kolom setelah perbaikan:", df.columns.tolist())
    return df

def create_preprocessor():
    categorical_features = ['category', 'subcategory', 'city', 'district', 'condition', 'type']
    text_feature = 'name'
    
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('text', TfidfVectorizer(max_features=100, lowercase=True), text_feature)
    ], remainder='drop')
    
    return preprocessor

if __name__ == "__main__":
    df = load_data('data/raw/dataset-sewa-360.csv')
    
    # Pilih fitur yang dibutuhkan
    required_cols = ['category', 'subcategory', 'name', 'city', 'district', 'condition', 'type']
    X = df[required_cols].copy()
    y = df['daily_price'].copy()

    # Buat dan latih preprocessor
    preprocessor = create_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    # Buat folder jika belum ada
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Simpan preprocessor dan data
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    pd.DataFrame(X_transformed).to_csv('data/processed/X_processed.csv', index=False)
    y.to_csv('data/processed/y_daily.csv', index=False)
    
    print("INFO: Preprocessing berhasil!")
    print(f"      Jumlah fitur: {X_transformed.shape[1]}")
    print(f"      Jumlah sampel: {X_transformed.shape[0]}")