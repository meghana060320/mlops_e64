import os, json, joblib, mlflow, mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from features import build_preprocessor, run_feature_pipeline

def train_and_register():
    p1 = run_feature_pipeline("kalamkari_drying_v1_normal.csv", "v1_normal")
    p2 = run_feature_pipeline("kalamkari_drying_v2_monsoon.csv", "v2_monsoon")
    df = pd.concat([pd.read_parquet(p1), pd.read_parquet(p2)], ignore_index=True)

    cols = [
        "fabric_material", "fabric_length_m", "num_dye_coats",
        "humidity_pct", "ambient_temp_c", "airflow_speed_ms",
        "rack_occupancy_pct", "moisture_load_index", "airflow_per_meter"
    ]
    X_train, X_test, y_train, y_test = train_test_split(df[cols], df["drying_duration_hrs"], test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Kalamkari_Drying_Time_Estimation")

    candidates = {
        "Ridge_Regression": Ridge(alpha=1.0),
        "Random_Forest_Regressor": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
    }

    best_name, best_rmse, best_pipe, best_run = None, float("inf"), None, None
    results = {}

    for name, est in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = Pipeline([("pre", build_preprocessor()), ("reg", est)]).fit(X_train, y_train)
            preds = np.maximum(0.0, pipe.predict(X_test))
            m = {
                "MAE": round(float(mean_absolute_error(y_test, preds)), 4),
                "RMSE": round(float(root_mean_squared_error(y_test, preds)), 4),
                "R2": round(float(r2_score(y_test, preds)), 4)
            }
            results[name] = m
            mlflow.log_params(est.get_params())
            mlflow.log_metrics(m)
            mlflow.sklearn.log_model(pipe, artifact_path="model")
            if m["RMSE"] < best_rmse:
                best_rmse, best_name, best_pipe, best_run = m["RMSE"], name, pipe, run.info.run_id

    mlflow.register_model(f"runs:/{best_run}/model", "KalamkariDryingTimeModel")
    joblib.dump(best_pipe, "best_model.joblib")
    with open("metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    print("Registered Best Model:", best_name, results)

if __name__ == "__main__":
    train_and_register()
