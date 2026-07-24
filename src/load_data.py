import pandas as pd
import numpy as np
import urllib.request
from pathlib import Path

def download_dataset_if_missing(file_path: Path):
    """
    Mengunduh dataset jika file tidak ditemukan di folder data/
    sehingga penguji yang melakukan clone dari nol dapat langsung mendapatkan data/
    tanpa error FileNotFoundError.
    """
    if file_path.exists():
        return

    print(f"File {file_path.name} belum ada di {file_path.parent}. Mengunduh dataset otomatis...")
    
    # URL cermin publik untuk dataset Car Price Prediction Kaggle
    url = "https://raw.githubusercontent.com/zafarali27/car-price-prediction/main/car_price_prediction_.csv"
    
    try:
        urllib.request.urlretrieve(url, file_path)
        print(f"Berhasil mengunduh dataset ke {file_path}")
    except Exception as e:
        print(f"Gagal mengunduh dari URL utama ({e}). Mencari alternatif local...")
        # Jika unduhan gagal, pastikan file lokal disalin jika tersedia di folder parent
        base_dir = file_path.parent.parent
        possible_sources = list(base_dir.glob("**/*.csv"))
        raw_sources = [f for f in possible_sources if "X_test" not in f.name and "y_test" not in f.name and f != file_path]
        if raw_sources:
            import shutil
            shutil.copy(raw_sources[0], file_path)
            print(f"Berhasil menyalin dataset dari {raw_sources[0]} ke {file_path}")
        else:
            raise FileNotFoundError(
                f"Dataset {file_path.name} tidak ditemukan dan gagal diunduh. Harap pastikan koneksi internet atau letakkan dataset di folder data/."
            )

def load_and_save_data():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)
    
    file_path = data_dir / "car_price_prediction_.csv"
    
    # Otomatis unduh jika penguji baru saja melalukan git clone
    download_dataset_if_missing(file_path)
            
    print(f"Membaca dataset mentah dari: {file_path}")
    df = pd.read_csv(file_path)
    
    print("\n" + "="*50)
    print("         INFORMASI DATASET MENTAH (TAHAP 1)")
    print("="*50)
    print(f"Jumlah baris (rows)   : {df.shape[0]}")
    print(f"Jumlah kolom (cols)   : {df.shape[1]}")
    print("\nTipe data tiap kolom:")
    print(df.dtypes)
    print("\nJumlah nilai hilang (missing values) per kolom:")
    print(df.isna().sum())
    print("\nLima baris pertama dataset:")
    print(df.head())
    print("="*50 + "\n")
    
    return df

if __name__ == "__main__":
    load_and_save_data()