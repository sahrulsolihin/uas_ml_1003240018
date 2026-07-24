import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def evaluate_model():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "model.joblib"
    report_dir = base_dir / "reports"
    report_dir.mkdir(exist_ok=True, parents=True)

    print(f"Mencari artefak model di: {model_path}")
    if not model_path.exists():
        print("Error: Model belum dilatih! Jalankan `python src/train.py` terlebih dahulu.")
        return

    # Memuat pipeline utuh dari joblib
    pipeline = joblib.load(model_path)

    # Membaca test set (hanya disentuh sekali di sini)
    X_test_path = base_dir / "data" / "X_test.csv"
    y_test_path = base_dir / "data" / "y_test.csv"
    
    if not X_test_path.exists() or not y_test_path.exists():
        print("Error: Test set tidak ditemukan! Jalankan `python src/train.py` terlebih dahulu.")
        return

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    # Melakukan prediksi pada test set
    y_pred = pipeline.predict(X_test)

    # Perhitungan Metrik Evaluasi
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 1. Grafik Actual vs Predicted -> reports/actual_vs_predicted.png
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='#2ca02c', edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal 1:1')
    plt.title('Evaluasi Test Set: Harga Aktual vs Harga Prediksi', fontsize=13, fontweight='bold')
    plt.xlabel('Harga Aktual ($)', fontsize=11)
    plt.ylabel('Harga Prediksi ($)', fontsize=11)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(report_dir / 'actual_vs_predicted.png', dpi=300)
    plt.close()

    # 2. Grafik Sebaran Error / Residuals -> reports/residuals_distribution.png
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color='#d62728', bins=30)
    plt.axvline(0, color='black', linestyle='--', lw=1.5)
    plt.title('Sebaran Error Prediksi (Residuals Distribution)', fontsize=13, fontweight='bold')
    plt.xlabel('Error (Actual - Predicted) ($)', fontsize=11)
    plt.ylabel('Frekuensi', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(report_dir / 'residuals_distribution.png', dpi=300)
    plt.close()

    # Output Ringkasan Evaluasi & Justifikasi Metrik Bisnis
    print("\n" + "="*60)
    print("        HASIL EVALUASI AKHIR PADA TEST SET (TAHAP 3)")
    print("="*60)
    print(f"MAE  (Mean Absolute Error)     : ${mae:,.2f}")
    print(f"RMSE (Root Mean Squared Error) : ${rmse:,.2f}")
    print(f"R2   (R-squared Score)         : {r2:.4f}")
    print(f"\nArtefak Grafik Evaluasi Tersimpan di:")
    print(f"  - {report_dir / 'actual_vs_predicted.png'}")
    print(f"  - {report_dir / 'residuals_distribution.png'}")
    print("\nJUSTIFIKASI METRIK BISNIS (KASUS B):")
    print("  1. MAE dipilih karena memberikan rata-rata selisih harga absolut yang mudah")
    print("     dijelaskan kepada pengguna/penjual marketplace dalam satuan Dolar ($).")
    print("  2. RMSE memberi bobot penalti lebih besar pada outlier error ekstrem (harga yang")
    print("     sangat jauh dari nilai wajar pasar).")
    print("  3. R2 menunjukkan persentase variansi harga yang mampu dijelaskan oleh model.")
    print("="*60 + "\n")

if __name__ == "__main__":
    evaluate_model()