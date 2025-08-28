# test_api.py
import requests
import json
from datetime import datetime

# Base URL API
BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, method="GET", data=None):
    """Helper function untuk test endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    print(f"\n{'='*60}")
    print(f"Testing: {method} {endpoint}")
    print(f"{'='*60}")
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

def main():
    """Test semua endpoint API"""
    
    print("🚀 Testing Klik Sewa ML API")
    print(f"Timestamp: {datetime.now()}")
    
    # Test 1: Health check
    test_endpoint("/health")
    
    # Test 2: Root endpoint
    test_endpoint("/")
    
    # Test 3: Model info
    test_endpoint("/model-info")
    
    # Test 4: Price recommendation (untuk Owner)
    owner_data = {
        "category": "Barang",
        "subcategory": "camping", 
        "name": "Kompor Portable + Gas",
        "city": "Bogor",
        "district": "Bogor Tengah",
        "condition": "baru",
        "type": "set"
    }
    test_endpoint("/recommend-price", "POST", owner_data)
    
    # Test 5: Price analysis (untuk Renter)
    renter_data = {
        "category": "Barang",
        "subcategory": "camping",
        "name": "Kompor Portable + Gas", 
        "city": "Bogor",
        "district": "Bogor Tengah",
        "condition": "baru",
        "type": "set",
        "current_price": 35000  # Harga yang akan dianalisis
    }
    test_endpoint("/analyze-price", "POST", renter_data)
    
    # Test 6: Price analysis dengan harga tinggi
    renter_data_expensive = {
        "category": "Barang",
        "subcategory": "camping",
        "name": "Kompor Portable + Gas",
        "city": "Bogor", 
        "district": "Bogor Tengah",
        "condition": "baru",
        "type": "set",
        "current_price": 50000  # Harga lebih tinggi
    }
    test_endpoint("/analyze-price", "POST", renter_data_expensive)
    
    # Test 7: Legacy predict endpoint
    test_endpoint("/predict", "POST", owner_data)
    
    # Test 8: Test dengan kategori berbeda
    electronics_data = {
        "category": "Barang",
        "subcategory": "elektronik",
        "name": "Kamera DSLR Canon",
        "city": "Bandung", 
        "district": "Bandung Wetan",
        "condition": "bekas",
        "type": "unit"
    }
    test_endpoint("/recommend-price", "POST", electronics_data)

if __name__ == "__main__":
    main()