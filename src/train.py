import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import joblib

def train_models():
    base_dir = Path(__file__).resolve().parent.parent
    file_path = base_dir / "data" / "car_price_prediction_.csv"
    if not file_path.exists():
        csv_files = list((base_dir / "data").glob("*.csv"))
        raw_csv = [f for f in csv_files if "X_test" not in f.name and "y_test" not in f.name]
        if raw_csv:
            file_path = raw_csv[0]

    df = pd.read_csv(file_path)

    # Feature engineering: Hitung umur kendaraan berdasarkan tahun 2026
    df['Car_Age'] = 2026 - df['Year']

    # Pisahkan fitur (X) dan target (y)
    # Hapus kolom ID unik ('Car ID'), 'Price' (target), dan 'Year' (redudan dengan Car_Age)
    drop_cols = ['Car ID', 'Price', 'Year']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['Price']

    # ATURAN WAJIB: Split train/test dilakukan SEBELUM preprocessing apa pun
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Simpan test set untuk evaluasi akhir (hanya disentuh sekali di evaluate.py)
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    X_test.to_csv(data_dir / "X_test.csv", index=False)
    y_test.to_csv(data_dir / "y_test.csv", index=False)

    # Identifikasi kolom numerik dan kategorikal
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Preprocessing Pipeline untuk Fitur Numerik & Kategorikal
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    # Definisi Minimal 3 Model Algoritma Regresi
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regressor": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_split=5, random_state=42
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, max_depth=8, random_state=42
        )
    }

    print("\n" + "="*60)
    print("      HASIL 5-FOLD CROSS VALIDATION PADA TRAIN SET (TAHAP 3)")
    print("="*60)

    results_summary = {}
    best_score = float('-inf')
    best_pipeline = None
    best_name = ""

    scoring = {
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error'
    }

    for name, model in models.items():
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        cv_results = cross_validate(
            full_pipeline, X_train, y_train, cv=5, scoring=scoring
        )

        r2_mean = cv_results['test_r2'].mean()
        r2_std = cv_results['test_r2'].std()
        mae_mean = -cv_results['test_mae'].mean()
        rmse_mean = -cv_results['test_rmse'].mean()

        results_summary[name] = {
            "r2_mean": float(r2_mean),
            "r2_std": float(r2_std),
            "mae_mean": float(mae_mean),
            "rmse_mean": float(rmse_mean)
        }

        print(f"[{name}]")
        print(f"  - R2 Score : {r2_mean:.4f} (+/- {r2_std:.4f})")
        print(f"  - MAE      : ${mae_mean:,.2f}")
        print(f"  - RMSE     : ${rmse_mean:,.2f}\n")

        # Pemilihan model terbaik berdasarkan nilai R2 CV
        if r2_mean > best_score:
            best_score = r2_mean
            best_pipeline = full_pipeline
            best_name = name

    # Fit pipeline model terbaik pada SELURUH data train
    best_pipeline.fit(X_train, y_train)

    # Simpan PIPELINE UTUH (bukan model telanjang) ke models/model.joblib
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, model_dir / "model.joblib")

    # Simpan metadata lengkap ke models/metadata.json
    metadata = {
        "best_model": best_name,
        "cv_r2_score": float(best_score),
        "cv_mae": results_summary[best_name]["mae_mean"],
        "cv_rmse": results_summary[best_name]["rmse_mean"],
        "all_models_cv": results_summary,
        "features": list(X.columns),
        "training_timestamp": datetime.now().isoformat()
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("="*60)
    print(f"Model Terbaik Terpilih: {best_name} (R2 CV = {best_score:.4f})")
    print(f"Pipeline Utuh Disimpan Ke: {model_dir / 'model.joblib'}")
    print(f"Metadata Disimpan Ke      : {model_dir / 'metadata.json'}")
    print("="*60 + "\n")

if __name__ == "__main__":
    train_models()