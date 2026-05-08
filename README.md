# GPTea — AI-Powered Tea Estate Management System

[![Python](https://img.shields.io/badge/Python-3.12.3-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

GPTea is an integrated AI-driven web application designed to support tea estate management in Sri Lanka. It combines three intelligent modules — tea yield forecasting, tea leaf disease detection, and an agricultural chatbot — into a single Streamlit-based platform built for real-world deployment.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Acknowledgements](#acknowledgements)
- [References](#references)

---

## Features

- **Tea Yield Prediction** — Monthly Yield Per Hectare (YPH) forecasting using Ridge Regression with walk-forward validation and lag feature engineering, trained on real estate data from the Dickoya region (2017–2025).
- **Tea Leaf Disease Detection** — Deep learning-based image classifier using MobileNetV2 and a custom leaf validator to identify common tea leaf diseases from uploaded photos.
- **Agricultural Chatbot** — Retrieval-Augmented Generation (RAG) chatbot backed by a sentence-transformer embedding model (all-MiniLM-L6-v2) and a domain-specific knowledge base, providing actionable agronomic advice.
- **Admin Dashboard** — Feedback collection and monitoring across all three modules.

---

## Project Structure

```
GPTea/
├── assets/                  # Images, icons, and profile photos
├── data/
│   ├── domain.csv           # Domain knowledge base for RAG chatbot
│   └── kb_embeddings.npy    # Precomputed sentence embeddings
├── feedback/
│   ├── chatbot_feedback.json
│   ├── disease_feedback.json
│   └── yield_feedback.json
├── models/
│   ├── all-MiniLM-L6-v2/        # Sentence transformer model
│   ├── Final_Ridge_Model_23_24_25.pkl  # Trained Ridge Regression model
│   ├── leaf_validator_best.keras       # Leaf validator (valid/invalid image)
│   ├── MobileNetV2_TeaDisease.keras    # Tea disease classifier
│   ├── class_names.json
│   └── model_registry.json
├── pages/
│   ├── 01_home.py
│   ├── 02_tea_yield.py
│   ├── 03_disease_detection.py
│   ├── 04_chatbot.py
│   ├── 05_admin.py
│   └── feedback_utils.py
├── utils/
│   ├── __init__.py
│   ├── chatbot_utils.py
│   ├── disease_utils.py
│   ├── rag_retrieval.py
│   └── tea_yield_model.py
├── app.py                   # Main Streamlit entry point
└── README.md
```

---

## Model Architecture

### 1. Tea Yield Prediction
- **Algorithm:** Ridge Regression (L2 regularization)
- **Validation:** Walk-forward validation with a 3-year rolling training window
- **Features (14 total):** Fertilizer inputs (T200, U709, Epsom Salt, Dolomite, Urea, BioShield), weather variables (temperature, rainfall, wind speed, ET₀), derived interaction feature (Dolomite × Moisture), and lag features (YPH_lag1, YPH_lag12, Compost_lag1)
- **Performance:** Test R² = 0.9261, RMSE = 6.23, MAE = 5.41 (2025 test set)
- **Explainability:** Model-intrinsic XAI via Ridge coefficient analysis — no SHAP or LIME required

### 2. Tea Leaf Disease Detection
- **Base Model:** MobileNetV2 (transfer learning)
- **Leaf Validator:** Custom Keras binary classifier to reject non-leaf images before disease classification
- **Framework:** TensorFlow / Keras

### 3. RAG Agricultural Chatbot
- **Embedding Model:** `all-MiniLM-L6-v2` (Sentence Transformers)
- **Knowledge Base:** Domain-specific tea agriculture CSV, embedded as `.npy` vectors
- **Retrieval:** Cosine similarity-based passage retrieval feeding into LLM response generation

---

## Installation

### Prerequisites

- Python 3.12.3
- `pip` package manager
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Kavishka2401/GPTea.git
cd GPTea

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The `models/` directory contains large model files. Ensure they are present before running the app. If they are not tracked by Git due to size, download them separately and place them in the `models/` folder as per the project structure above.

---

## How to Run

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

Navigate between modules using the sidebar:
- **Home** — Overview and introduction
- **Tea Yield Prediction** — Enter fertilizer and weather data to forecast monthly YPH
- **Disease Detection** — Upload a tea leaf image for disease classification
- **Chatbot** — Ask agronomic questions powered by the RAG pipeline
- **Admin** — View collected feedback across modules

---

## Acknowledgements

- **Ms. Ayeshka Jayasundara** — Supervisor, Informatics Institute of Technology
- **Mr. D.M. Sriyan Dissanayake** — Domain expertise and access to historical estate records, Dickoya region
- **Mr. Indika Hiran Wijesinghe** — Deputy Director IT, Sri Lanka Tea Board — domain and technical evaluation
- **Dr. Sahan Priyanayana** — Technical review of system architecture and ML components
- **Informatics Institute of Technology & Robert Gordon University Aberdeen** — Academic and institutional support

Weather data sourced from the [Open-Meteo Archive API](https://archive-api.open-meteo.com/v1/archive).



> *This project was developed as a second-year research project at the Informatics Institute of Technology, Colombo, Sri Lanka.*