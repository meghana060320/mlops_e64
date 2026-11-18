# Kalamkari_Drying_Time

End-to-end MLOps pipeline for estimating drying duration (in hours) for hand-printed Kalamkari fabrics near Machilipatnam.

## Repository Files
- `Kalamkari_Drying_Time.ipynb`: Complete experiment notebook
- `kalamkari_drying_v1_normal.csv` & `kalamkari_drying_v2_monsoon.csv`: Dataset versions (normal vs. 75–95% monsoon humidity)
- `features.py`: Automated preprocessing & Parquet feature store (`moisture_load_index`, `airflow_per_meter`)
- `train.py`: Ridge vs. Random Forest training & MLflow model registry (`KalamkariDryingTimeModel`)
- `app.py`: FastAPI `/predict` inference endpoint
- `test_app.py` & `.github/workflows/ci.yml`: Automated GitHub Actions tests
- `Kalamkari_Drying_Time_Report.pdf`: Full implementation report

## Quickstart
```bash
pip install -r requirements.txt
python train.py
pytest -v test_app.py
uvicorn app:app --reload
```
