# 🚕 NYC Taxi Trip Duration Prediction — MLOps Pipeline

An end-to-end MLOps project that predicts NYC green taxi trip durations using a machine learning model, built with industry-standard MLOps tools.

---

## 📌 Project Overview

This project builds a complete ML pipeline for predicting taxi trip durations in New York City using the [NYC TLC Green Taxi dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). It covers the full MLOps lifecycle — from data exploration to model deployment and monitoring.

---

## 🏗️ Architecture
Raw Data (Parquet)
↓
Data Exploration & Feature Engineering
↓
Model Training (Linear Regression, Lasso, Ridge)
↓
Experiment Tracking (MLflow)
↓
Model Registry (MLflow)
↓
Model Deployment (Flask + Docker)
↓
Monitoring (Evidently AI + PostgreSQL + Grafana)
↓
CI/CD (GitHub Actions + Pytest)

```
nyc-taxi-trip-analysis/
│
├── 📓 Notebooks/
│   ├── 01_data_exploration.ipynb       # EDA and feature engineering
│   ├── 02_mlflow_tracking.ipynb        # Experiment tracking and model registry
│   ├── 03_deployment.ipynb             # Flask API + Docker deployment
│   ├── 04_monitoring.ipynb             # Evidently + Grafana monitoring
│   └── 05_best_practices.ipynb         # Testing + CI/CD
│
├── 📁 data/
│   ├── green_tripdata_2026-01.parquet  # January training data
│   └── green_tripdata_2026-02.parquet  # February validation data
│
├── 📁 models/
│   └── lin_reg.bin                     # Pickled DictVectorizer + model
│
├── 📁 tests/
│   └── test_predict.py                 # Unit tests
│
├── 📁 .github/workflows/
│   └── ci.yml                          # GitHub Actions CI/CD
│
├── 🐍 predict.py                       # Flask prediction service
├── 🐳 Dockerfile                       # Docker image for prediction service
├── 🐳 docker-compose.yml               # PostgreSQL + Grafana stack
├── 📄 requirements.txt                 # Python dependencies
└── 📄 README.md
```
---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data Processing | Pandas, NumPy, PyArrow |
| ML Training | Scikit-learn (Linear Regression, Lasso, Ridge) |
| Experiment Tracking | MLflow |
| Model Serving | Flask, Docker |
| Monitoring | Evidently AI, PostgreSQL, Grafana |
| Testing | Pytest |
| CI/CD | GitHub Actions |
| Environment | GitHub Codespaces, Jupyter Lab |

---

## 📊 Dataset

- **Source:** NYC TLC Trip Record Data
- **Type:** Green Taxi Trips
- **Period:** January 2026 (training) & February 2026 (validation)
- **Size:** ~40,000 trips per month after filtering

---

## 🔧 Setup & Installation

### Prerequisites
- GitHub Codespaces or Linux environment
- Docker and Docker Compose
- Conda or pip

### 1. Clone the repository
```bash
git clone https://github.com/NikitaJagtap39/nyc-taxi-trip-analysis.git
cd nyc-taxi-trip-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the data
Download NYC Green Taxi parquet files from the [TLC website](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and place them in the `data/` folder.

---

## 🚀 Running the Project

### Train the model
Open and run `01_data_exploration.ipynb` and `02_mlflow_tracking.ipynb` in Jupyter Lab.

### Start MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Run the prediction service
```bash
docker build -t nyc-taxi-duration:v1 .
docker run -d -p 9696:9696 nyc-taxi-duration:v1
```

### Test the API
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID": 65, "DOLocationID": 233, "trip_distance": 6.2}'
```

Expected response:
```json
{"duration_minutes": 23.9}
```

### Start monitoring stack
```bash
docker-compose up -d
```
Then open Grafana at `http://localhost:3000` (admin/admin)

---

## 🧪 Running Tests
```bash
pytest tests/test_predict.py -v
```

---

## 📈 Model Performance

| Model | RMSE Train | RMSE Val |
|-------|-----------|---------|
| Linear Regression | 8.05 | 8.34 |
| Ridge | 8.02 | 8.34 |
| Lasso | 9.80 | 9.86 |

**Best model:** Ridge Regression (RMSE: 8.34 minutes on validation set)

---

## 📉 Monitoring

The monitoring pipeline tracks:
- **Data drift** between training (January) and production (February) data
- **RMSE** over time
- All metrics stored in PostgreSQL and visualized in Grafana

---

## 🔄 CI/CD

GitHub Actions automatically runs all unit tests on every push to `main`. The pipeline:
1. Sets up Python 3.12
2. Installs dependencies
3. Runs pytest

---

## 📚 Acknowledgements

This project follows the curriculum of the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) by [DataTalks.Club](https://datatalks.club/).
