import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

FEATURE_STORE_DIR = "feature_store"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["moisture_load_index"] = data["fabric_length_m"] * data["num_dye_coats"] * (data["humidity_pct"] / 100.0)
    data["airflow_per_meter"] = data["airflow_speed_ms"] / np.maximum(data["fabric_length_m"], 0.1)
    return data

def build_preprocessor() -> ColumnTransformer:
    num_cols = [
        "fabric_length_m", "num_dye_coats", "humidity_pct",
        "ambient_temp_c", "airflow_speed_ms", "rack_occupancy_pct",
        "moisture_load_index", "airflow_per_meter"
    ]
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["fabric_material"])
    ])

def run_feature_pipeline(csv_path: str, tag: str) -> str:
    os.makedirs(FEATURE_STORE_DIR, exist_ok=True)
    df_feat = engineer_features(pd.read_csv(csv_path))
    out_path = os.path.join(FEATURE_STORE_DIR, f"features_{tag}.parquet")
    df_feat.to_parquet(out_path, index=False)
    return out_path
