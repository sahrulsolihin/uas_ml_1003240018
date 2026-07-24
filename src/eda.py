import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def run_eda():
    base_dir = Path(__file__).resolve().parent.parent
    file_path = base_dir / "data" / "car_price_prediction_.csv"
    if not file_path.exists():
        csv_files = list((base_dir / "data").glob("*.csv"))
        raw_csv = [f for f in csv_files if "X_test" not in f.name and "y_test" not in f.name]
        if raw_csv:
            file_path = raw_csv[0]

    report_dir = base_dir / "reports"
    report_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(file_path)

    # Menghitung Car_Age untuk EDA
    df['Car_Age'] = 2026 - df['Year']

    # 1. Grafik Sebaran Target (Histogram & KDE) -> target_distribution.png
    plt.figure(figsize=(9, 5))
    sns.histplot(df['Price'], kde=True, color='#1f77b4', bins=30)
    plt.title('Distribusi Sebaran Harga Mobil Bekas (Target)', fontsize=14, fontweight='bold')
    plt.xlabel('Price ($)', fontsize=12)
    plt.ylabel('Frekuensi', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(report_dir / 'target_distribution.png', dpi=300)
    plt.close()

    # 2. Grafik Nilai Hilang per Kolom -> missing_values.png
    plt.figure(figsize=(9, 5))
    missing_data = df.isna().sum()
    bars = plt.bar(missing_data.index, missing_data.values, color='#e377c2')
    plt.title('Pemeriksaan Nilai Hilang (Missing Values) per Kolom', fontsize=14, fontweight='bold')
    plt.ylabel('Jumlah Nilai Hilang', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(report_dir / 'missing_values.png', dpi=300)
    plt.close()

    # 3. Correlation Heatmap & Scatter Plot -> feature_vs_target.png
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap korelasi numerik
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0], cbar=True)
    axes[0].set_title('Heatmap Korelasi Fitur Numerik vs Target', fontweight='bold')

    # Scatter plot Mileage vs Price dengan hue Condition
    sns.scatterplot(x='Mileage', y='Price', hue='Condition', data=df, alpha=0.6, ax=axes[1], palette='tab10')
    axes[1].set_title('Hubungan Mileage vs Price berdasarkan Condition', fontweight='bold')
    axes[1].set_xlabel('Mileage (KM)')
    axes[1].set_ylabel('Price ($)')

    plt.tight_layout()
    plt.savefig(report_dir / 'feature_vs_target.png', dpi=300)
    plt.close()

    # 4. Boxplot Deteksi Outlier Harga -> outlier_check.png
    plt.figure(figsize=(9, 5))
    sns.boxplot(x='Brand', y='Price', data=df, palette='Set3')
    plt.title('Boxplot Distribusi Harga per Brand (Deteksi Outlier Ekstrem)', fontsize=14, fontweight='bold')
    plt.xlabel('Brand Mobil', fontsize=12)
    plt.ylabel('Harga ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(report_dir / 'outlier_check.png', dpi=300)
    plt.close()

    # Pelaporan temuan EDA di terminal
    num_duplicates = df.duplicated().sum()
    print("="*60)
    print("                HASIL ANALISIS EDA & TEMUAN (TAHAP 2)")
    print("="*60)
    print(f"1. Grafik tersimpan ke folder: {report_dir}")
    print("   - target_distribution.png (Sebaran Target Harga)")
    print("   - missing_values.png      (Visualisasi Nilai Hilang)")
    print("   - feature_vs_target.png   (Heatmap Korelasi & Scatter Mileage vs Price)")
    print("   - outlier_check.png       (Boxplot Harga per Brand)")
    print(f"\n2. Deteksi Baris Duplikat: {num_duplicates} baris duplikat ditemukan.")
    print("\n3. TIGA Kekotoran Data Nyata yang Ditemukan:")
    print("   a. ID Unik 'Car ID': Kolom identifier tanpa nilai prediktif (harus dibuang sebelum modeling).")
    print("   b. Hubungan Non-Linear & Outlier: Hubungan antara Umur (Car_Age) dan Mileage terhadap Price tidak linier murni.")
    print("   c. Kolom 'Year' redundant setelah dibuat 'Car_Age' (harus dibuang 'Year' untuk menghindari collinearity).")
    print("\n4. Tiga Prakiraan Sebelum Modeling:")
    print("   - Prakiraan 1: Fitur Mileage dan Car_Age akan menjadi prediktor utama harga kendaraan.")
    print("   - Prakiraan 2: Model regresi non-linear (Random Forest / Gradient Boosting) akan mengalahkan Linear Regression.")
    print("   - Prakiraan 3: Kategori 'Condition' dan 'Brand' akan memberikan bobot offset yang signifikan terhadap harga dasar.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_eda()