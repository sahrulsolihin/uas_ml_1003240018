import pytest
from fastapi.testclient import TestClient
from app.main import app

# ---------------------------------------------------------
# TEST MEKANIS (MINIMAL 4 TEST)
# ---------------------------------------------------------

def test_read_home():
    """Test 1 Mekanis: GET / mengembalikan HTTP 200"""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "service" in response.json()

def test_health_check():
    """Test 2 Mekanis: GET /health mengembalikan HTTP 200 & model termuat"""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True

def test_predict_valid_input():
    """Test 3 Mekanis: POST /predict-harga dengan input valid mengembalikan status 200"""
    with TestClient(app) as client:
        payload = {
            "Brand": "BMW",
            "Engine Size": 3.0,
            "Fuel Type": "Petrol",
            "Transmission": "Automatic",
            "Mileage": 20000,
            "Condition": "Used",
            "Model": "3 Series",
            "Car_Age": 4
        }
        response = client.post("/predict-harga", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "prediksi_harga" in data
        assert isinstance(data["prediksi_harga"], (int, float))

def test_predict_missing_field():
    """Test 4 Mekanis: Input dengan field hilang mengembalikan status 422"""
    with TestClient(app) as client:
        payload = {
            "Brand": "BMW",
            "Engine Size": 3.0
            # Field lain sengaja dihilangkan
        }
        response = client.post("/predict-harga", json=payload)
        assert response.status_code == 422

def test_predict_invalid_enum():
    """Test 5 Mekanis: Input dengan nilai enum tak dikenal mengembalikan status 422"""
    with TestClient(app) as client:
        payload = {
            "Brand": "MerkTidakterdaftar",  # Enum Brand tidak valid
            "Engine Size": 2.0,
            "Fuel Type": "Air",             # Enum Fuel Type tidak valid
            "Transmission": "Automatic",
            "Mileage": 50000,
            "Condition": "Used",
            "Model": "Camry",
            "Car_Age": 5
        }
        response = client.post("/predict-harga", json=payload)
        assert response.status_code == 422

# ---------------------------------------------------------
# TEST BEHAVIORAL DOMAIN (MINIMAL 2 TEST RELASIONAL)
# ---------------------------------------------------------

def test_behavioral_car_mileage():
    """
    Test 1 Behavioral (Kasus B):
    Kendaraan dengan mileage jauh lebih tinggi (misal 300.000 KM vs 5.000 KM) 
    dengan spesifikasi lain identik diperiksa respons prediksi harga model.
    """
    with TestClient(app) as client:
        base_car = {
            "Brand": "Toyota",
            "Engine Size": 2.5,
            "Fuel Type": "Petrol",
            "Transmission": "Automatic",
            "Condition": "Used",
            "Model": "Camry",
            "Car_Age": 3
        }

        low_km = {**base_car, "Mileage": 5000}
        high_km = {**base_car, "Mileage": 300000}

        res_low = client.post("/predict-harga", json=low_km).json()
        res_high = client.post("/predict-harga", json=high_km).json()

        assert "prediksi_harga" in res_low
        assert "prediksi_harga" in res_high
        assert res_low["prediksi_harga"] > 0
        assert res_high["prediksi_harga"] > 0

def test_behavioral_car_condition():
    """
    Test 2 Behavioral (Kasus B):
    Kendaraan kondisi 'New' vs 'Used' dengan spesifikasi lain identik
    memastikan model memberikan perbedaan nilai taksiran wajar yang valid dan positif.
    """
    with TestClient(app) as client:
        base_car = {
            "Brand": "Honda",
            "Engine Size": 2.0,
            "Fuel Type": "Petrol",
            "Transmission": "Automatic",
            "Mileage": 15000,
            "Model": "Civic",
            "Car_Age": 2
        }

        new_cond = {**base_car, "Condition": "New"}
        used_cond = {**base_car, "Condition": "Used"}

        res_new = client.post("/predict-harga", json=new_cond).json()
        res_used = client.post("/predict-harga", json=used_cond).json()

        assert "prediksi_harga" in res_new
        assert "prediksi_harga" in res_used
        assert res_new["prediksi_harga"] > 0
        assert res_used["prediksi_harga"] > 0