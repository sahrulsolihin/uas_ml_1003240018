# Projek UAS Machine Learning End-to-End

**Nama Mahasiswa**: Sahrul Solihin  
**NIM**: 1003240018  
**Mata Kuliah**: Machine Learning End-to-End (Semester Ganjil 2026/2027)  
**Institut Teknologi Tangerang Selatan (ITTS)**  
**Pilihan Kasus**: Kasus B — Regresi: Estimasi Harga Kendaraan Bekas

---

## 1. Deskripsi Masalah

Marketplace otomotif ingin membantu penjual kendaraan bekas untuk mendapatkan rekomendasi harga jual yang wajar dan objektif berdasarkan spesifikasi kendaraan. Target prediksi adalah harga kontinu (`Price` dalam satuan USD). 

Tantangan utama yang ditangani pada sistem ini meliputi:
- Pemodelan hubungan non-linear antara umur kendaraan (`Car_Age`) dan kilometer penggunaan (`Mileage`) terhadap harga.
- Penanganan outlier harga ekstrem dan transformasi data kategorikal (`Brand`, `Fuel Type`, `Transmission`, `Condition`).

---

## 2. Sumber & Lisensi Data

- **Sumber Dataset**: Kaggle — [Car Price Prediction Dataset](https://www.kaggle.com/datasets/zafarali27/car-price-prediction?resource=download)
- **Lisensi Data**: CC0: Public Domain / Open Data License
- **Jumlah Data**: ~2.500 baris dengan fitur kendaraan bekas.

---

## 3. Versi Lingkungan & Package Utama

Proyek ini dibangun dan diuji menggunakan versi package sebagai berikut:
- **Python**: `3.13.1`
- **pandas**: `2.2.3`
- **scikit-learn**: `1.6.1`
- **fastapi**: `0.115.11`
- **uvicorn**: `0.34.0`
- **pydantic**: `2.10.6`
- **joblib**: `1.4.2`
- **pytest**: `8.3.5`

---

## 4. Langkah-Langkah Menjalankan Proyek dari Nol

### Step 1: Clone Repositori & Buat Virtual Environment
```bash
git clone https://github.com/sahrulsolihin/uas-ml-1003240018.git
cd uas-ml-1003240018

# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# Atau Linux/macOS: source venv/bin/activate
```

### Step 2: Install Dependensi Training
```bash
pip install -r requirements.txt
```

### Step 3: Jalankan Alur End-to-End (Data -> EDA -> Train -> Evaluate)
```bash
# 1. Memuat & periksa data mentah
python src/load_data.py

# 2. Jalankan Analisis Data Eksploratif (Grafik tersimpan ke reports/)
python src/eda.py

# 3. Latih & bandingkan model ML (Artefak tersimpan ke models/)
python src/train.py

# 4. Evaluasi model pada test set & buat grafik evaluasi
python src/evaluate.py
```

### Step 4: Jalankan Server API FastAPI
```bash
uvicorn app.main:app --reload
```
Akses dokumentasi Swagger UI interaktif di browser: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Step 5: Jalankan Automated Test (pytest)
```bash
python -m pytest tests/ -v
```

---

## 5. Penjelasan Artefak & `.gitignore`

> **Mengapa `data/` dan `models/` Masuk `.gitignore`?**  
> Folder `data/` (dataset mentah & split test) dan `models/` (file biner `.joblib` & metadata) tidak dikomit ke Git repositori untuk menjaga ukuran repositori tetap ringan dan menghindari penyimpanan file artefak biner yang sering berubah. Penguji dapat dengan mudah mereproduksi ulang seluruh isi `data/` dan `models/` dari nol hanya dengan mengisolasi repositori dan menjalankan skrip `python src/load_data.py` lalu `python src/train.py`.

---

## 6. Penjelasan Lingkungan Serving & Requirements

> **Mengapa `requirements-api.txt` Menggunakan Versi Terkunci (Pinned Version `==`)?**  
> Lingkungan serving REST API di lingkungan produksi memerlukan versi dependensi yang di-pin persis (`requirements-api.txt`) untuk menjamin stabilitas (*reproducibility*), mencegah timbulnya *breaking changes* akibat *update* versi pustaka di masa mendatang, serta memastikan skema serialisasi model `joblib` selalu kompatibel. Sebaliknya, lingkungan training (`requirements.txt`) menggunakan versi yang lebih fleksibel (`>=`) untuk memungkinkan penguji atau pengembang mencoba fitur pustaka terbaru selama eksperimen.

---

## 7. Contoh Pemanggilan API (cURL Request & Response)

### A. Contoh Request Prediksi BERHASIL (HTTP Status 200)

**Command cURL**:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict-harga' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Brand": "BMW",
  "Engine Size": 3.0,
  "Fuel Type": "Petrol",
  "Transmission": "Automatic",
  "Mileage": 25000,
  "Condition": "Used",
  "Model": "3 Series",
  "Car_Age": 4
}'
```

**Respons JSON (200 OK)**:
```json
{
  "status": "success",
  "prediksi_harga": 42150.75,
  "currency": "USD",
  "input_summary": {
    "Brand": "BMW",
    "Model": "3 Series",
    "Car_Age": 4,
    "Mileage": 25000,
    "Condition": "Used"
  }
}
```

---

### B. Contoh Request TIDAK VALID (HTTP Status 422 - Validation Error)

**Command cURL**:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict-harga' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Brand": "MerkTidakTerdaftar",
  "Engine Size": 3.0,
  "Fuel Type": "Petrol"
}'
```

**Respons JSON (422 Unprocessable Entity)**:
```json
{
  "detail": [
    {
      "type": "enum",
      "loc": ["body", "Brand"],
      "msg": "Input should be 'Tesla', 'BMW', 'Audi', 'Ford', 'Mercedes', 'Honda' or 'Toyota'",
      "input": "MerkTidakTerdaftar"
    },
    {
      "type": "missing",
      "loc": ["body", "Transmission"],
      "msg": "Field required",
      "input": { ... }
    }
  ]
}
```

---

## 8. Struktur Folder Proyek

```
uas-ml-1003240018/
├── src/
│   ├── load_data.py      # Pemuatan data mentah & verifikasi statistik
│   ├── eda.py            # Analisis Data Eksploratif (menghasilkan 4 grafik PNG)
│   ├── train.py          # Preprocessing Pipeline, CV 5-Fold & Training Model
│   └── evaluate.py       # Evaluasi Test Set & Pembuatan Grafik Evaluasi
├── app/
│   └── main.py           # REST API FastAPI (Lifespan, Pydantic Enum & Logging)
├── tests/
│   └── test_api.py       # Test Otomatis Pytest (5 Mekanis & 2 Behavioral)
├── data/                 # Dataset mentah & split test set (masuk .gitignore)
├── models/               # Artefak model.joblib & metadata.json (masuk .gitignore)
├── reports/              # Grafik EDA & Evaluasi PNG (dikomit ke repo)
├── requirements.txt      # Dependensi Lingkungan Training
├── requirements-api.txt  # Dependensi Lingkungan Serving (versi di-pin persis)
├── .gitignore            # Pengatur pengabaian file Git
└── README.md             # Dokumentasi lengkap proyek
```
