# run.py
import subprocess
import sys
import os

def run_step(script):
    print(f"\n[RUNNING] {script}")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("[ERROR]", result.stderr)
    if result.returncode != 0:
        print("[FAILED] Proses gagal.")
        exit(1)

if __name__ == "__main__":
    print("[START] Mulai Pipeline Klik Sewa ML Backend\n")
    
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

    print("\n[SUCCESS] SEMUA PROSES SELESAI!")
    print("[INFO] Jalankan API: uvicorn src.api.main:app --reload")
    print("[INFO] Laporan: folder reports/")