import pytest
import pandas as pd
from fastapi.testclient import TestClient
from features import engineer_features
from train import train_and_register
from app import app

@pytest.fixture(scope="session", autouse=True)
def setup():
    train_and_register()

client = TestClient(app)

def test_feature_engineering():
    df = engineer_features(pd.DataFrame([{
        "fabric_material": "Cotton", "fabric_length_m": 20.0, "num_dye_coats": 3,
        "humidity_pct": 80.0, "ambient_temp_c": 28.0, "airflow_speed_ms": 2.0, "rack_occupancy_pct": 50.0
    }]))
    assert df["moisture_load_index"].iloc[0] == pytest.approx(48.0)
    assert df["airflow_per_meter"].iloc[0] == pytest.approx(0.1)

def test_non_negative_prediction():
    res = client.post("/predict", json={
        "fabric_material": "Silk", "fabric_length_m": 15.0, "num_dye_coats": 2,
        "humidity_pct": 55.0, "ambient_temp_c": 34.0, "airflow_speed_ms": 3.5, "rack_occupancy_pct": 30.0
    })
    assert res.status_code == 200
    assert res.json()["estimated_drying_duration_hrs"] >= 0.0

def test_invalid_input():
    res = client.post("/predict", json={
        "fabric_material": "Nylon", "fabric_length_m": -5.0, "num_dye_coats": 2,
        "humidity_pct": 55.0, "ambient_temp_c": 34.0, "airflow_speed_ms": 3.5, "rack_occupancy_pct": 30.0
    })
    assert res.status_code == 422
