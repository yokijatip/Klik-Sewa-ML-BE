# 🏠 Klik Sewa ML Backend - Smart Price Recommendation System

Sistem rekomendasi harga berbasis Machine Learning untuk aplikasi Klik Sewa menggunakan algoritma **Random Forest** dan **Gradient Boosting** dengan akurasi tinggi untuk mendukung ekosistem sewa-menyewa yang fair dan efisien.

## 📋 Table of Contents

- [Overview](#overview)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation Metrics](#evaluation-metrics)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Examples](#usage-examples)
- [Performance Results](#performance-results)
- [Technical Implementation](#technical-implementation)
- [For Academic Purpose](#for-academic-purpose)

## 🎯 Overview

**Klik Sewa ML Backend** adalah sistem backend berbasis FastAPI yang mengintegrasikan model machine learning untuk memberikan rekomendasi harga sewa harian yang akurat dan fair. Sistem ini melayani dua use case utama:

1. **Owner Journey**: Mendapatkan rekomendasi harga optimal saat memasukkan produk
2. **Renter Journey**: Menganalisis keadilan harga produk yang akan disewa

### Key Features
- ✅ **Dual Algorithm Support**: Random Forest & Gradient Boosting
- ✅ **Smart Price Analysis**: Deteksi harga fair vs overpriced
- ✅ **Comprehensive Market Analysis**: Range harga, posisi kompetitif
- ✅ **Real-time Prediction**: Response time < 100ms
- ✅ **Model Performance Tracking**: MAE, R², MAPE monitoring
- ✅ **RESTful API**: Standard HTTP API dengan dokumentasi lengkap

## 🤖 Machine Learning Models

### 1. Random Forest Regressor
**Random Forest** adalah ensemble method yang menggabungkan multiple decision trees untuk prediksi yang lebih robust.

**Kelebihan:**
- Mengurangi overfitting dibanding single decision tree
- Dapat menangani missing values dan outliers
- Memberikan feature importance secara otomatis
- Robust terhadap noise dalam data

**Hyperparameters:**
```python
RandomForestRegressor(
    n_estimators=150,      # Jumlah decision trees
    max_depth=20,          # Kedalaman maksimum tree
    min_samples_split=5,   # Minimum samples untuk split
    min_samples_leaf=2,    # Minimum samples di leaf node
    random_state=42
)
```

### 2. Gradient Boosting Regressor
**Gradient Boosting** adalah ensemble method yang membangun model secara sequential, dimana setiap model memperbaiki error dari model sebelumnya.

**Kelebihan:**
- High predictive accuracy
- Dapat menangani berbagai tipe data
- Built-in feature selection
- Robust terhadap outliers

**Hyperparameters:**
```python
GradientBoostingRegressor(
    n_estimators=150,      # Jumlah boosting stages
    max_depth=6,           # Kedalaman maksimum setiap tree
    learning_rate=0.1,     # Learning rate untuk shrinkage
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

## 📊 Evaluation Metrics

### 1. **R² Score (Coefficient of Determination)**
**Definisi:** Mengukur proporsi varians dalam variabel dependen yang dapat diprediksi dari variabel independen.

**Formula:** R² = 1 - (SS_res / SS_tot)

**Interpretasi:**
- **1.0**: Prediksi sempurna
- **0.8-0.9**: Sangat baik
- **0.6-0.8**: Baik
- **< 0.6**: Perlu perbaikan

**Hasil Project:** R² = 0.9536 ✅ (Sangat Baik)

### 2. **MAE (Mean Absolute Error)**
**Definisi:** Rata-rata dari nilai absolut error antara prediksi dan nilai aktual.

**Formula:** MAE = (1/n) × Σ|y_actual - y_predicted|

**Interpretasi:**
- Dalam rupiah, menunjukkan rata-rata selisih prediksi
- Semakin kecil semakin baik
- Tidak sensitif terhadap outliers

**Hasil Project:** MAE = Rp 61,870 ✅

### 3. **RMSE (Root Mean Square Error)**
**Definisi:** Akar kuadrat dari rata-rata kuadrat error.

**Formula:** RMSE = √[(1/n) × Σ(y_actual - y_predicted)²]

**Interpretasi:**
- Memberikan penalti lebih besar untuk error yang besar
- Dalam satuan yang sama dengan target variable
- Sensitif terhadap outliers

**Hasil Project:** RMSE = Rp 117,155 ✅

### 4. **MAPE (Mean Absolute Percentage Error)**
**Definisi:** Rata-rata persentase error absolut.

**Formula:** MAPE = (100/n) × Σ|((y_actual - y_predicted) / y_actual)|

**Interpretasi:**
- **< 10%**: Sangat akurat
- **10-20%**: Akurat
- **20-50%**: Cukup akurat
- **> 50%**: Kurang akurat

**Hasil Project:** MAPE = 17.02% ✅ (Akurat)

### 5. **Accuracy (±20% Threshold)**
**Definisi:** Persentase prediksi yang berada dalam threshold ±20% dari nilai aktual.

**Formula:** Accuracy = (Jumlah prediksi dalam threshold / Total prediksi) × 100%

**Hasil Project:** 72.2% ✅

## 🛠 API Endpoints

### 1. `/recommend-price` (Owner Endpoint)
**Purpose:** Memberikan rekomendasi harga untuk owner yang ingin memasukkan produk.

**Method:** `POST`

**Request Body:**
```json
{
  "category": "Barang",
  "subcategory": "camping", 
  "name": "Kompor Portable + Gas",
  "city": "Bogor",
  "district": "Bogor Tengah",
  "condition": "baru",
  "type": "set"
}
```

**Response:**
```json
{
  "recommended_price_daily": 185000,
  "market_analysis": {
    "price_range_rp": "Rp 157,250 - Rp 212,750",
    "market_average_rp": "Rp 178,000",
    "competitive_position": "Above Average"
  },
  "model_information": {
    "algorithm": "Random Forest",
    "model_accuracy": 92.3,
    "confidence_score": 87.5,
    "training_data_points": 15847
  },
  "analysis_factors": [
    "Product category and condition",
    "Geographic location (Bandung, West Java)",
    "Seasonal demand patterns",
    "Similar product pricing history"
  ],
  "confidence_score": 0.875,
  "timestamp": "2024-08-28T10:30:00"
}
```

### 2. `/analyze-price` (Renter Endpoint)
**Purpose:** Menganalisis keadilan harga produk untuk renter.

**Method:** `POST`

**Request Body:**
```json
{
  "category": "Barang",
  "subcategory": "camping",
  "name": "Kompor Portable + Gas",
  "city": "Bogor",
  "district": "Bogor Tengah", 
  "condition": "baru",
  "type": "set",
  "current_price": 35000
}
```

**Response:**
```json
{
  "current_price": 35000,
  "recommended_price": 30000,
  "price_analysis": {
    "price_difference": 5000,
    "price_difference_percent": 16.7,
    "status_detail": "Harga 16.7% lebih tinggi dari rekomendasi",
    "market_comparison": "Above Market Average",
    "is_using_recommendation": false,
    "recommended_price_rp": "Rp 30,000",
    "current_price_rp": "Rp 35,000"
  },
  "recommendation_status": "Above Recommendation",
  "confidence_score": 0.875,
  "timestamp": "2024-08-28T10:30:00"
}
```

### 3. Additional Endpoints

- **`GET /health`** - Health check
- **`GET /model-info`** - Model information dan metrics
- **`POST /predict`** - Legacy endpoint (backward compatibility)

## 📁 Project Structure

```
klik-sewa-ml-be/
├── 📂 data/
│   ├── raw/                    # Dataset asli (CSV)
│   └── processed/              # Data setelah preprocessing
├── 📂 models/                  # Model ML dan preprocessor
│   ├── best_model.pkl         # Model terbaik (Random Forest/GB)
│   ├── preprocessor.pkl       # OneHotEncoder + TF-IDF
│   ├── model_metrics.json     # Performance metrics
│   └── feature_importance.json
├── 📂 src/
│   ├── 📂 api/
│   │   └── main.py           # FastAPI application
│   ├── preprocessing.py       # Data preprocessing pipeline
│   ├── train_model.py        # Model training dan evaluation  
│   └── visualize.py          # Visualisasi dan reporting
├── 📂 reports/                # Visualisasi hasil (PNG files)
├── 📂 utils/
├── run.py                    # Pipeline automation script
├── test_api.py              # API testing script
├── requirements.txt         # Python dependencies  
└── README.md               # Dokumentasi lengkap
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip atau conda
- Git

### 1. Clone Repository
```bash
git clone <repository-url>
cd klik-sewa-ml-be
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

# Using conda
conda create -n klik-sewa python=3.11
conda activate klik-sewa
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset
Pastikan file `data/raw/dataset-sewa-360.csv` tersedia dengan struktur:
```csv
id,category,subcategory,name,city,district,condition,daily_price,weekly_price,monthly_price,deposit,unit
```

### 5. Run Complete Pipeline
```bash
python run.py
```

### 6. Start API Server
```bash
uvicorn src.api.main:app --reload
```

### 7. Access API Documentation
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 💡 Usage Examples

### Python Client Example
```python
import requests

# Owner mendapatkan rekomendasi harga
owner_request = {
    "category": "Barang",
    "subcategory": "elektronik",
    "name": "Kamera DSLR Canon EOS",
    "city": "Bandung", 
    "district": "Bandung Wetan",
    "condition": "bekas",
    "type": "unit"
}

response = requests.post(
    "http://localhost:8000/recommend-price", 
    json=owner_request
)
print(f"Rekomendasi harga: Rp {response.json()['recommended_price_daily']:,}")

# Renter menganalisis harga
renter_request = {
    **owner_request,
    "current_price": 150000
}

response = requests.post(
    "http://localhost:8000/analyze-price",
    json=renter_request  
)
print(f"Status harga: {response.json()['recommendation_status']}")
```

### cURL Examples
```bash
# Get price recommendation
curl -X POST "http://localhost:8000/recommend-price" \
  -H "Content-Type: application/json" \
  -d '{
    "category": "Barang",
    "subcategory": "camping",
    "name": "Tenda Dome 4 Orang",
    "city": "Bogor",
    "district": "Bogor Selatan", 
    "condition": "baru",
    "type": "unit"
  }'

# Analyze existing price
curl -X POST "http://localhost:8000/analyze-price" \
  -H "Content-Type: application/json" \
  -d '{
    "category": "Barang",
    "subcategory": "camping",
    "name": "Tenda Dome 4 Orang",
    "city": "Bogor",
    "district": "Bogor Selatan",
    "condition": "baru", 
    "type": "unit",
    "current_price": 75000
  }'
```

## 📈 Performance Results

### Model Comparison
| Metric | Random Forest | Gradient Boosting | Winner |
|--------|---------------|-------------------|--------|
| **R² Score** | **0.9536** | 0.9043 | 🏆 Random Forest |
| **MAE** | **Rp 61,870** | Rp 80,640 | 🏆 Random Forest |
| **RMSE** | **Rp 117,155** | Rp 168,333 | 🏆 Random Forest |  
| **MAPE** | **17.02%** | 30.07% | 🏆 Random Forest |
| **Accuracy (±20%)** | **72.2%** | 61.1% | 🏆 Random Forest |

### Dataset Statistics
- **Total Samples:** 360 products
- **Training Set:** 288 samples (80%)
- **Test Set:** 72 samples (20%)
- **Features:** 152 (after preprocessing)
- **Categories:** Barang, Kendaraan, dll.
- **Price Range:** Rp 10,000 - Rp 500,000/day

### Feature Engineering
- **Categorical Encoding:** OneHotEncoder untuk category, city, condition, dll.
- **Text Processing:** TF-IDF untuk product names (max 100 features)
- **Geographic Features:** City dan district encoding
- **Condition Mapping:** baru, bekas, rusak mapping

## 🔧 Technical Implementation

### Data Preprocessing Pipeline
```python
# Categorical features
categorical_features = ['category', 'subcategory', 'city', 'district', 'condition', 'type']

# Text feature  
text_feature = 'name'

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('text', TfidfVectorizer(max_features=100), text_feature)
])
```

### Model Selection Logic
```python
# Automatic model selection berdasarkan R² score
if rf_r2_score >= gb_r2_score:
    best_model = random_forest_model
    model_name = "RandomForest"
else:
    best_model = gradient_boosting_model  
    model_name = "GradientBoosting"
```

### API Response Format
- **Consistent Structure:** Semua endpoint menggunakan struktur response yang konsisten
- **Error Handling:** Comprehensive error handling dengan HTTP status codes
- **Timestamp:** Semua response include timestamp untuk auditing
- **Validation:** Pydantic models untuk request/response validation

## 🎓 For Academic Purpose

### Thesis/Research Context
Proyek ini cocok untuk:
- **Skripsi Machine Learning:** Implementasi ensemble methods dalam domain e-commerce
- **Penelitian Price Prediction:** Studi kasus pricing strategy dalam marketplace
- **Computer Science Final Project:** Full-stack ML application dengan API integration

### Academic Contributions
1. **Comparative Analysis:** Random Forest vs Gradient Boosting untuk price prediction
2. **Feature Engineering:** Kombinasi categorical dan text features dalam rental marketplace  
3. **Business Application:** Real-world implementation untuk dual user journey (Owner/Renter)
4. **Performance Evaluation:** Comprehensive metrics evaluation dengan multiple criteria

### Research Findings
- **Random Forest superior** untuk dataset rental marketplace (R² = 95.36%)
- **Text features penting** (product name) untuk akurasi prediksi harga
- **Geographic features** memberikan kontribusi signifikan untuk pricing regional
- **Ensemble methods robust** terhadap variasi kategori produk

### Future Improvements
- [ ] Deep Learning models (Neural Networks) 
- [ ] Time series analysis untuk seasonal pricing
- [ ] Collaborative filtering untuk personalized recommendations
- [ ] A/B testing framework untuk model performance monitoring
- [ ] Real-time model retraining pipeline

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`) 
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Developer:** [Your Name]  
**Email:** [your.email@example.com]  
**LinkedIn:** [your-linkedin-profile]  
**University:** [Your University Name]

---

**⭐ Star this repository jika membantu project skripsi Anda!**

> *"Smart pricing for fair rental marketplace - Powered by Machine Learning"*