# AI-Powered Inventory Purchase Forecasting System

An end-to-end machine learning pipeline designed to optimize purchasing decisions, cut down inventory waste, and bridge the gap between traditional time-series methods and advanced machine learning models. 

## Project Overview
This production-oriented framework scales demand forecasting by combining statistical modeling (`ARIMA`/`ARIMAX`) with powerful gradient-boosted trees (`XGBoost`, `LightGBM`, `CatBoost`) and ensemble regressors. It includes robust time-based splitting to prevent data leakage and handles post-model explainability using SHAP values.

## Repository Structure
```text
├── data/                  # Local data storage (Gitignored)
├── src/
│   ├── data_loader.py     # Data ingestion and robust cleaning
│   ├── features.py        # Feature engineering (lags, rolling averages)
│   ├── models.py          # Model architecture & evaluation orchestration
│   └── explain.py         # SHAP value interpretability generator
├── main.py                # Pipeline execution entry point
└── requirements.txt       # Production dependencies
