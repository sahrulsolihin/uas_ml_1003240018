import pandas as pd
import numpy as np
from pathlib import Path

def load_and_save_data():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)
    
    file_path = data_dir / "car_price_prediction_.csv"
    
    # Memastikan file dataset ada di folder data/
    if not file_path.exists():
        print(f"File {file_path} tidak ditemukan. Memeriksa file alternatif di data/...")
        csv_files = list(data_dir.glob("*.csv"))
        raw_csv = [f for f in csv_files if "X_test" not in f.name and "y_test" not in f.name]
        if raw_csv:
            file_path = raw_csv[0]
            print(f"Menggunakan file dataset: {file_path.name}")
        else:
            raise FileNotFoundError(
                f"Dataset tidak ditemukan di {data_dir}. Harap letakkan file CSV dataset kendaraan di folder data/."
            )
            
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