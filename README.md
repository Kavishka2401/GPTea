# GPTea — C_1_Yield_Prediction Branch

[![Python](https://img.shields.io/badge/Python-3.12.3-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red.svg)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-Ridge%20Regression-green.svg)]()

This branch contains the complete **Tea Yield Prediction** component of the GPTea system — from raw data processing and model training through to the deployed Streamlit web interface. It is structured into three folders: the machine learning pipeline (Jupyter notebooks), the final trained model, and the website (Streamlit page).

---

## Table of Contents

- [Overview](#overview)
- [Branch Structure](#branch-structure)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Final Model](#final-model)
- [Website](#website)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [Model Performance](#model-performance)
- [Explainability](#explainability)
- [Installation](#installation)
- [How to Run](#how-to-run)

---

## Overview

This module forecasts monthly tea yield (kg/hectare) for a tea estate in the Dickoya region of Sri Lanka. It combines fertilizer application records with historical weather data from the Open-Meteo Archive API, using a temporally-aware Ridge Regression model trained on real estate-level data from 2017 to 2025.

Every prediction is accompanied by a feature-level explanation using Ridge coefficient analysis, giving estate managers clear, actionable insights into what is driving each forecast.

---

## Branch Structure

```
C_1_Yield_Prediction/
│
├── Machine_Learning_Pipeline/
│   ├── 01_Data_Wrangling.ipynb                  # Data loading, cleaning & merging
│   ├── Exploratory_Data_Analysis.ipynb           # EDA with scatter plots & visualizations
│   ├── Preprocessing_and_ModelTraining.ipynb     # Feature engineering & model experiments
│   ├── Final_Model_Preprocessing_Training.ipynb  # Final Ridge model training & saving
│   ├── XAI.ipynb                                 # Explainability testing with LLM
│   └── README.md
│
├── final-model/
│   ├── Final_Ridge_Model_23_24_25.pkl            # Trained Ridge Regression model
│   └── README.md
│
├── website/
│   ├── tea-yield-prediction-page.py              # Streamlit yield prediction UI
│   └── README.md
│
└── README.md
```

---

## Machine Learning Pipeline

The `Machine_Learning_Pipeline/` folder contains the full experimental workflow as Jupyter notebooks, intended to be run in order:

| Notebook | Purpose |
|---|---|
| `01_Data_Wrangling.ipynb` | Loads estate yield & fertilizer records, retrieves weather data from Open-Meteo API, aggregates daily weather to monthly, and merges all sources into a single modelling dataset (96 monthly observations) |
| `Exploratory_Data_Analysis.ipynb` | Visualizes feature distributions, scatter plots, correlations, and seasonal yield patterns to guide feature engineering decisions |
| `Preprocessing_and_ModelTraining.ipynb` | Engineers lag and interaction features, experiments with multiple models (Ridge, LASSO, CatBoost, XGBoost, Random Forest, Neural Network) under walk-forward validation |
| `Final_Model_Preprocessing_Training.ipynb` | Retrains the final Ridge Regression model on the 2023–2025 window, saves the fitted model as a `.pkl` file to `final-model/` |
| `XAI.ipynb` | Tests Ridge coefficient-based feature contribution explanations and LLM-generated natural language summaries for predictions |

---

## Final Model

The `final-model/` folder stores the production-ready trained model:

| File | Description |
|---|---|
| `Final_Ridge_Model_23_24_25.pkl` | Ridge Regression model trained on the 2023–2025 rolling window — the model loaded by the Streamlit web application at inference time |

> **Note:** This `.pkl` file is loaded directly by `tea-yield-prediction-page.py` to serve predictions. Do not retrain unless updating the rolling window with new estate data.

---

## Website

The `website/` folder contains the Streamlit frontend for the yield prediction module:

| File | Description |
|---|---|
| `tea-yield-prediction-page.py` | Streamlit page that accepts monthly fertilizer and weather inputs, loads the trained Ridge model, generates a YPH forecast, and displays a ranked feature contribution explanation table alongside a downloadable report |

---

## Model Architecture

- **Algorithm:** Ridge Regression (L2 regularization)
- **Validation Strategy:** Walk-forward validation with a 3-year rolling training window
- **Training Data:** Monthly estate records, Dickoya region, Sri Lanka (2017–2025) — 96 observations
- **Weather Source:** [Open-Meteo Archive API](https://archive-api.open-meteo.com/v1/archive)
- **Framework:** scikit-learn

### Why Ridge Regression?
Ridge Regression was selected after empirical comparison with CatBoost, XGBoost, Random Forest, Neural Networks, and LASSO under identical walk-forward evaluation. It avoids overfitting on the small-sample temporal dataset, preserves all 14 features for agronomic interpretability (unlike LASSO which zeroed out 11 of 14), and produces stable, conservative predictions well-suited to agricultural planning.

---

## Features

The model uses **14 input features** across three categories:

### Fertilizer Inputs
| Feature | Description |
|---|---|
| T200 | T200 fertilizer application (kg) |
| U709 | U709 fertilizer application (kg) |
| Epsom Salt | Epsom Salt application (kg) |
| Dolomite | Dolomite application (kg) |
| Urea | Urea application (kg) |
| BioShield | BioShield application (kg) |

### Weather Variables
| Feature | Description |
|---|---|
| temperature_2m_mean | Monthly mean temperature (°C) |
| rain_sum | Monthly total rainfall (mm) |
| wind_speed_10m_mean | Monthly mean wind speed (km/h) |
| ET₀ | Reference evapotranspiration — FAO-56 Penman-Monteith (mm/day) |

### Engineered Features
| Feature | Description |
|---|---|
| Dolomite_moisture | Interaction: Dolomite × (rain_sum − ET₀), capturing synergy between amendment and soil moisture |
| YPH_lag1 | Yield from the previous month |
| YPH_lag12 | Yield from the same month in the prior year (seasonal anchor) |
| Compost_lag1 | Compost application from the previous month (delayed nutrient uptake) |

---

## Model Performance

Evaluated using walk-forward validation (3-year rolling window) across all test folds:

| Metric | Value |
|---|---|
| Test R² | 0.9261 |
| RMSE | 6.23 kg/ha |
| MAE | 5.41 kg/ha |

The model captures over **92% of variance** in monthly tea yield. The slight conservative under-prediction tendency is a deliberate effect of Ridge regularization — underestimating yield is preferable to overestimating in agricultural resource planning.

### Comparison with Other Models (Walk-Forward Validation)

| Model | Test R² | Notes |
|---|---|---|
| **Ridge Regression** | **0.9201** | Best balanced generalization |
| LASSO | 0.9687* | Degenerate — zeroes out 11/14 features |
| CatBoost | 0.8313 | Overfits within narrow 3-year window |
| XGBoost | 0.7855 | Overfits within narrow 3-year window |
| Neural Network | 0.8045 | High variance across folds |
| Random Forest | 0.3342 | Collapses under small-sample constraints |

*LASSO's high R² is misleading — it reduces to a near-autoregressive predictor with no actionable fertilizer or weather insights.

---

## Explainability

This module uses **model-intrinsic XAI** via direct Ridge coefficient analysis — no SHAP, no LIME, no external libraries required. For each prediction, the signed contribution of every feature is computed as:

```
contribution_i = x_i (scaled) × β_i
```

This produces a ranked explanation table showing which features drove the prediction up or down and by how much. All 14 features retain non-zero contributions, enabling complete agronomic explanations for every forecast.

The `XAI.ipynb` notebook further experiments with passing this contribution table to an LLM to generate human-readable natural language summaries for estate managers.

Key findings:
- **YPH_lag1** and **YPH_lag12** are consistently the strongest predictors
- **rain_sum** and **ET₀** are the most influential weather variables
- **Dolomite** and **Urea** show consistent positive contributions
- **Dolomite_moisture** captures synergistic effects neither variable captures independently

---

## Installation

### Prerequisites

- Python 3.12.3
- `pip` package manager

### Steps

```bash
# 1. Clone the repository and switch to this branch
git clone https://github.com/Kavishka2401/GPTea.git
cd GPTea
git checkout C_1_Yield_Prediction

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Streamlit Web App
```bash
cd website
streamlit run tea-yield-prediction-page.py
```
Opens at `http://localhost:8501`. Enter monthly fertilizer and weather values, click **Predict**, and view the forecasted YPH with a feature contribution breakdown and downloadable report.

### Jupyter Notebooks (ML Pipeline)
```bash
cd Machine_Learning_Pipeline
jupyter notebook
```
Run notebooks in order: `01_Data_Wrangling` → `Exploratory_Data_Analysis` → `Preprocessing_and_ModelTraining` → `Final_Model_Preprocessing_Training` → `XAI`.

---
