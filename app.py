import os, joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from features import engineer_features

app = FastAPI(title="Kalamkari Drying Time Estimator")

class BatchInput(BaseModel):
    fabric_material: str = Field(..., examples=["Cotton"])
    fabric_length_m: float = Field(..., gt=0, examples=[25.0])
    num_dye_coats: int = Field(..., ge=1, examples=[3])
    humidity_pct: float = Field(..., ge=0, le=100, examples=[85.0])
    ambient_temp_c: float = Field(..., ge=0, le=60, examples=[27.0])
    airflow_speed_ms: float = Field(..., ge=0, examples=[1.5])
    rack_occupancy_pct: float = Field(..., ge=0, le=100, examples=[70.0])

@app.get("/")
def root():
    return {"message": "Kalamkari Drying Time API is running"}

@app.post("/predict")
def predict(batch: BatchInput):
    if batch.fabric_material not in ["Cotton", "Silk"]:
        raise HTTPException(status_code=422, detail="fabric_material must be Cotton or Silk")
    df = engineer_features(pd.DataFrame([batch.model_dump()]))
    model = joblib.load("best_model.joblib")
    pred = round(max(0.0, float(model.predict(df)[0])), 2)
    return {
        "estimated_drying_duration_hrs": pred,
        "moisture_load_index": round(float(df["moisture_load_index"].iloc[0]), 4),
        "airflow_per_meter": round(float(df["airflow_per_meter"].iloc[0]), 4)
    }
