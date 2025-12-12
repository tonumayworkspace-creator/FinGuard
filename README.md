# FinGuard --- Credit Card Fraud Detection System

A practical Data Science + AI Engineering project showcasing end-to-end
ML development, model evaluation, and a user-friendly prediction
interface.

## 🔍 Project Overview

FinGuard is a full-stack fraud detection system designed to identify
suspicious credit card transactions in real time. This project
demonstrates the complete lifecycle of a Data Science & AI Engineering
workflow: - Data preprocessing and feature handling - Addressing class
imbalance - Model training and statistical evaluation - A compact,
interactive Flask-based UI - A prediction API suitable for integration -
Versioned model artifacts and reproducibility practices

## 📁 Dataset

Dataset file used: `creditcard-database.csv`

## 🧠 Machine Learning Pipeline

### 1. Preprocessing

-   Stratified train-test split
-   Numeric conversion + simple validation
-   Support for future scaling and transformation pipelines

### 2. Model Training

-   Logistic Regression baseline
-   Handles class imbalance using class_weight='balanced'

### 3. Evaluation

-   ROC AUC
-   Precision, Recall, F1
-   Full classification report

## 📊 Model Artifacts

Stored in:

    models/
     ├── fin_guard_model.pkl
     └── model_metrics.json

## 🖥️ Application Features

-   Dataset preview
-   Model training UI
-   Prediction form with hints, autofill, confirmation modal
-   `/predict_api` endpoint

## 🧩 Tech Stack

Python, Flask, scikit-learn, Pandas, NumPy, HTML/CSS/JS

## 📦 Project Structure

    FinGuard/
    │── app/
    │── data/
    │── models/
    │── requirements.txt
    └── README.md

## 🚀 Running the Project

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python app/app.py

## 👤 Author

Tonumay Bhattacharya
