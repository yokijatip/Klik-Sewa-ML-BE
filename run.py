# run.py
import subprocess
import sys
import os

def run_step(script):
    print(f"\n🚀 Menjalankan: {script}")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("❌ Error:", result.stderr)
    if result.returncode != 0:
        print("🛑 Proses gagal.")
        exit(1)

if __name__ == "__main__":
    print("🚀 Mulai Pipeline Klik Sewa ML Backend\n")
    
    # Pastikan folder ada
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1. Preprocessing
    run_step('src/preprocessing.py')

    # 2. Training
    run_step('src/train_model.py')

    # 3. Visualisasi
    os.makedirs('reports', exist_ok=True)
    run_step('src/visualize.py')

    print("\n✅ SEMUA PROSES SELESAI!")
    print("🔥 Jalankan API: uvicorn src.api.main:app --reload")
    print("📊 Laporan: folder reports/")