import logging
from enum import Enum
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
import pandas as pd

# Setting up logging for predictions (Persyaratan Rubrik Tahap 4: Logging Prediksi Berjalan)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ML_API")

# Define Enums for strict Pydantic validation (Types, Ranges, and Enums)
class BrandEnum(str, Enum):
    Tesla = "Tesla"
    BMW = "BMW"
    Audi = "Audi"
    Ford = "Ford"
    Mercedes = "Mercedes"
    Honda = "Honda"
    Toyota = "Toyota"

class FuelTypeEnum(str, Enum):
    Petrol = "Petrol"
    Diesel = "Diesel"
    Electric = "Electric"
    Hybrid = "Hybrid"

class TransmissionEnum(str, Enum):
    Manual = "Manual"
    Automatic = "Automatic"

class ConditionEnum(str, Enum):
    New = "New"
    Used = "Used"
    Like_New = "Like New"

ml_models: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load trained model pipeline in lifespan manager
    try:
        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / "models" / "model.joblib"
        if not model_path.exists():
            model_path = Path.cwd() / "models" / "model.joblib"

        if model_path.exists():
            ml_models["pipeline"] = joblib.load(model_path)
            logger.info(f"Model pipeline successfully loaded from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model pipeline: {e}")
    
    yield
    ml_models.clear()
    logger.info("Cleared loaded models from memory.")

app = FastAPI(
    title="API Estimasi Harga Kendaraan Bekas - ITTS",
    description="API Machine Learning End-to-End untuk estimasi harga jual wajar mobil bekas",
    version="1.0.0",
    lifespan=lifespan
)

class CarInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Brand: BrandEnum
    Engine_Size: float = Field(..., alias="Engine Size", ge=0.5, le=10.0)
    Fuel_Type: FuelTypeEnum = Field(..., alias="Fuel Type")
    Transmission: TransmissionEnum
    Mileage: int = Field(..., ge=0, le=1000000)
    Condition: ConditionEnum
    Model: str = Field(..., min_length=1)
    Car_Age: int = Field(..., ge=0, le=50)

@app.get("/")
def home():
    return {
        "service": "API Estimasi Harga Kendaraan Bekas ITTS",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "predict": "/predict-harga (POST)"
        }
    }

@app.get("/health")
def health_check():
    if "pipeline" not in ml_models:
        try:
            model_path = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
            if model_path.exists():
                ml_models["pipeline"] = joblib.load(model_path)
        except Exception:
            pass

    model_loaded = "pipeline" in ml_models
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }

@app.post("/predict-harga")
def predict_harga(data: CarInput):
    if "pipeline" not in ml_models:
        try:
            model_path = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
            if model_path.exists():
                ml_models["pipeline"] = joblib.load(model_path)
        except Exception:
            pass

    if "pipeline" not in ml_models:
        logger.error("Prediction request failed: Model pipeline is not loaded.")
        raise HTTPException(status_code=500, detail="Model belum dimuat di server.")

    # Convert Pydantic input to DataFrame with exact column names expected by pipeline
    input_data = data.model_dump(by_alias=True)
    input_df = pd.DataFrame([input_data])

    try:
        prediction = ml_models["pipeline"].predict(input_df)
        predicted_price = round(float(prediction[0]), 2)

        # Logging prediksi berjalan
        logger.info(f"PREDICTION SUCCESS | Input: {input_data} | Predicted Price: ${predicted_price}")

        return {
            "status": "success",
            "prediksi_harga": predicted_price,
            "currency": "USD",
            "input_summary": {
                "Brand": data.Brand.value,
                "Model": data.Model,
                "Car_Age": data.Car_Age,
                "Mileage": data.Mileage,
                "Condition": data.Condition.value
            }
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=422, detail=f"Error saat inferensi model: {str(e)}")