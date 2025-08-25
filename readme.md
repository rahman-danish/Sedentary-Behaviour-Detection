# Sedentary Behaviour Detection using Fitbit Data
MSc Dissertation — Danish Rahman, Ulster University (2025)

## Overview
Detects sedentary behaviour in office settings using Fitbit data. Compares ML (KNN, LR, RF, SVM, XGBoost) and DL (Dense NN, LSTM). Includes a Streamlit app for CSV upload, predictions, and SHAP explanations.

## Project structure
- `data_preprocessing/` — cleaning/merging scripts
- `models/` — ML & DL training/evaluation
- `app/` — Streamlit app
- `results/` — figures, confusion matrices, SHAP plots, (optional) small TensorBoard logs

## How to run
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
