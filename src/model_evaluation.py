import pandas as pd
import numpy as np
import logging
import joblib
import os
import json
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from utils import logger_add

# ✅ Import dvclive
from dvclive import Live

logger = logger_add("logs", "model_evaluation")


def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    logger.debug('Parameters retrieved from %s', params_path)
    return params


def load_data(path):
    data = pd.read_csv(path)
    logger.debug("DATA LOADED")
    return data


def load_model(path):
    model = joblib.load(path)
    logger.debug("MODEL LOADED FROM %s", path)
    return model


def save_metrics(metrics: dict, report_dir="REPORT"):
    os.makedirs(report_dir, exist_ok=True)
    save_path = os.path.join(report_dir, "evaluation_metrics.json")
    with open(save_path, 'w') as f:
        json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metrics.items()}, f, indent=4)
    logger.debug(f"METRICS SAVED SUCCESSFULLY TO {save_path}")


def evaluate_model(model, test_data: pd.DataFrame, live: Live):
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)

    logger.debug(f"MODEL EVALUATION METRICS: acc={accuracy:.4f}, prec={precision:.4f}, rec={recall:.4f}, f1={f1:.4f}")

    # 🔹 Log metrics to dvclive Live
    live.log("accuracy", accuracy)
    live.log("precision", precision)
    live.log("recall", recall)
    live.log("f1_score", f1)
    live.next_step()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }


def main():
    # Initialize dvclive Live in DVC experiment mode
    live = Live("dvclive", save_dvc_exp=True)  # ✅ logs will be saved to DVC experiments automatically

    params = load_params("params.yaml")
    test_data = load_data("./process/featured/test.csv")
    model = load_model("./models/model.pkl")

    metrics = evaluate_model(model, test_data, live)

    save_metrics(metrics)
    logger.debug("MODEL EVALUATION COMPLETED")


if __name__ == "__main__":
    main()

