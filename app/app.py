# app/app.py
# FinGuard — Flask app with model training, metrics, ROC, SHAP and fixed shap_index paths.
# Complete file — overwrite your existing app/app.py with this.

import os
import json
import pandas as pd
from flask import Flask, render_template, request, url_for, redirect, send_from_directory, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import joblib
from datetime import datetime
import traceback
import numpy as np

# SHAP and plotting
import matplotlib
matplotlib.use("Agg")
import shap
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "fin_guard_dev_secret"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard-database.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "fin_guard_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
ROC_DATA_PATH = os.path.join(MODELS_DIR, "roc_data.json")
SHAP_INDEX_PATH = os.path.join(MODELS_DIR, "shap_index.json")
SHAP_HTML_DIR = os.path.join(MODELS_DIR, "shap_html")  # directory to store per-instance SHAP html


def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SHAP_HTML_DIR, exist_ok=True)


def load_feature_columns():
    if not os.path.exists(DATA_PATH):
        return []
    try:
        df = pd.read_csv(DATA_PATH, nrows=1)
    except Exception:
        return []
    return [c for c in df.columns if c != "Class"]


def safe_joblib_load(path):
    try:
        return True, joblib.load(path)
    except Exception:
        return False, traceback.format_exc()


@app.route("/")
def index():
    dataset_exists = os.path.exists(DATA_PATH)
    model_exists = os.path.exists(MODEL_PATH)
    return render_template("index.html", dataset_exists=dataset_exists, model_exists=model_exists, data_path=DATA_PATH)


@app.route("/preview")
def preview():
    if not os.path.exists(DATA_PATH):
        return render_template("error.html", message=f"Dataset not found at {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH, nrows=10)
    except Exception as e:
        return render_template("error.html", message=str(e))
    return render_template("preview.html", columns=list(df.columns), rows=df.fillna("").values.tolist(), row_count=len(df))


@app.route("/train")
def train():
    """
    Train pipeline, save model, compute ROC data, generate SHAP per-instance HTML artifacts.
    """
    if not os.path.exists(DATA_PATH):
        return render_template("error.html", message="Dataset not found.")

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        return render_template("error.html", message=f"Error loading dataset: {e}")

    if "Class" not in df.columns:
        return render_template("error.html", message="Dataset must contain a 'Class' column.")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # convert to numeric and fill NaN
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        return render_template("error.html", message=f"Training failed: {e}")

    # Save model
    try:
        ensure_models_dir()
        joblib.dump(pipeline, MODEL_PATH)
    except Exception as e:
        return render_template("error.html", message=f"Failed to save model: {e}")

    # Evaluate
    try:
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        roc_auc = float(roc_auc_score(y_test, probs)) if probs is not None else None
        report_text = classification_report(y_test, preds)
    except Exception as e:
        return render_template("error.html", message=f"Failed to evaluate model: {e}")

    # Save ROC data
    try:
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist(), "roc_auc": roc_auc}
        with open(ROC_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(roc_data, f)
    except Exception:
        pass

    # Save metrics JSON
    metrics = {"timestamp": datetime.utcnow().isoformat() + "Z", "roc_auc": roc_auc, "classification_report": report_text}
    try:
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass

    # Create SHAP per-instance interactive HTML files (sample up to N instances)
    try:
        ensure_models_dir()
        bg = X_train.sample(n=min(200, len(X_train)), random_state=42)
        explainer = shap.Explainer(pipeline.predict_proba, bg, algorithm="auto")

        n_instances = min(20, len(X_test))
        sample_instances = X_test.reset_index(drop=True).sample(n=n_instances, random_state=42)

        shap_index = []
        for i, (_, row) in enumerate(sample_instances.iterrows()):
            try:
                instance_df = row.to_frame().T
                shap_vals = explainer(instance_df)

                fname = f"shap_instance_{i}.html"
                outpath = os.path.join(SHAP_HTML_DIR, fname)
                # IMPORTANT: Save file path relative to MODELS_DIR (not PROJECT_ROOT) so the template can request '/models/<relpath>'
                try:
                    shap.save_html(outpath, shap_vals)
                except Exception:
                    try:
                        fplot = shap.plots.force(shap_vals[0], matplotlib=False)
                        shap.save_html(outpath, fplot)
                    except Exception:
                        continue

                # create relative file path with forward slashes, relative to MODELS_DIR
                rel_to_models = os.path.relpath(outpath, MODELS_DIR).replace("\\", "/")

                # small preview of up to 10 features for UI
                preview_obj = {}
                for j, col in enumerate(instance_df.columns[:10]):
                    try:
                        v = instance_df.iloc[0, j]
                        preview_obj[col] = float(v) if np.isfinite(v) else None
                    except Exception:
                        preview_obj[col] = None

                meta = {
                    "id": i,
                    "file": rel_to_models,   # e.g. "shap_html/shap_instance_0.html"
                    "preview": preview_obj
                }
                shap_index.append(meta)
            except Exception:
                continue

        # save index file (overwrite)
        try:
            with open(SHAP_INDEX_PATH, "w", encoding="utf-8") as f:
                json.dump(shap_index, f, indent=2)
        except Exception:
            pass

    except Exception:
        # On SHAP failure continue - training still succeeded
        pass

    return render_template("train_result.html", report=report_text, roc_auc=roc_auc, model_path=MODEL_PATH)


@app.route("/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        return render_template("error.html", message="No metrics found. Please run /train first.")
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as e:
        return render_template("error.html", message=f"Failed to read metrics: {e}")
    return render_template("metrics.html", metrics=metrics)


@app.route("/roc")
def roc():
    if not os.path.exists(ROC_DATA_PATH):
        return render_template("error.html", message="No ROC data available. Run /train first to generate ROC data."), 404
    try:
        with open(ROC_DATA_PATH, "r", encoding="utf-8") as f:
            roc = json.load(f)
    except Exception as e:
        return render_template("error.html", message=f"Failed to read ROC data: {e}")
    trace = {"fpr": roc.get("fpr", []), "tpr": roc.get("tpr", []), "roc_auc": roc.get("roc_auc", None)}
    return render_template("roc.html", roc_json=json.dumps(trace))


@app.route("/shap")
def shap_view():
    if not os.path.exists(SHAP_INDEX_PATH):
        return render_template("error.html", message="No SHAP instances available. Train model to generate SHAP explainers and instance HTML files."), 404
    try:
        with open(SHAP_INDEX_PATH, "r", encoding="utf-8") as f:
            idx = json.load(f)
    except Exception as e:
        return render_template("error.html", message=f"Failed to read SHAP index: {e}")
    return render_template("shap.html", instances=idx)


@app.route("/models/<path:filename>")
def models_static(filename):
    # Serve files located under models/ directory
    return send_from_directory(MODELS_DIR, filename, as_attachment=False)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    feature_cols = load_feature_columns()
    if request.method == "GET":
        return render_template("predict.html", feature_cols=feature_cols)

    ok, model_or_err = safe_joblib_load(MODEL_PATH)
    if not ok:
        msg = "Model not available or failed to load. Please retrain the model by using the Train action.\n\nDetails:\n" + str(model_or_err)
        return render_template("error.html", message=msg)

    model = model_or_err
    input_data = {}
    for col in feature_cols:
        raw = (request.form.get(col) or "").strip()
        try:
            input_data[col] = float(raw) if raw != "" else 0.0
        except:
            input_data[col] = 0.0

    df_input = pd.DataFrame([input_data])
    try:
        pred = int(model.predict(df_input)[0])
        probs = model.predict_proba(df_input)[0] if hasattr(model, "predict_proba") else None
        prob_map = {}
        if probs is not None:
            try:
                classes = model.named_steps["model"].classes_
                for cls, p in zip(classes, probs):
                    prob_map[str(cls)] = float(p)
            except Exception:
                prob_map = {"0": float(probs[0]), "1": float(probs[1])}
    except Exception as e:
        return render_template("error.html", message=f"Prediction failed: {e}")

    return render_template("prediction_result.html", predicted=pred, prob_map=prob_map, input_df=input_data)


@app.route("/predict_api", methods=["POST"])
def predict_api():
    ok, model_or_err = safe_joblib_load(MODEL_PATH)
    if not ok:
        return jsonify({"error": "Model not available. Please run /train."}), 400
    model = model_or_err

    data = request.get_json(force=True)
    if not data or "features" not in data:
        return jsonify({"error": "JSON must contain top-level 'features' field."}), 400

    try:
        df_input = pd.DataFrame([data["features"]])
    except Exception as e:
        return jsonify({"error": f"Invalid features format: {e}"}), 400

    try:
        pred = int(model.predict(df_input)[0])
        probs = model.predict_proba(df_input)[0] if hasattr(model, "predict_proba") else None
        prob_map = {}
        if probs is not None:
            try:
                classes = model.named_steps["model"].classes_
                for cls, p in zip(classes, probs):
                    prob_map[str(cls)] = float(p)
            except Exception:
                prob_map = {"0": float(probs[0]), "1": float(probs[1])}
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify({"prediction": pred, "probabilities": prob_map}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
