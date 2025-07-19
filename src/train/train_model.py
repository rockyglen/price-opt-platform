import os
import joblib
import pandas as pd
import numpy as np
import subprocess
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

import mlflow
import mlflow.sklearn

# Load environment
load_dotenv()

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")
MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "pricing-ml-model")

if not TRACKING_URI:
    raise EnvironmentError("MLFLOW_TRACKING_URI is not set in .env")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Paths
DATA_PATH = "data/raw/interactions.csv"
FEATURE_ORDER_PATH = "src/model/feature_order.pkl"

# Load and preprocess data
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["interaction_id", "customer_id", "product_id", "converted"])
X = pd.get_dummies(X, columns=["category", "segment", "time_of_day", "day_of_week", "season"])
y = df["converted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=42))
])

pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# Save feature order
os.makedirs(os.path.dirname(FEATURE_ORDER_PATH), exist_ok=True)
joblib.dump(X.columns.tolist(), FEATURE_ORDER_PATH)

# Start MLflow run
with mlflow.start_run():
    mlflow.log_param("model", "logistic_regression")
    mlflow.log_param("scaling", True)
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("train_rows", len(X_train))

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc", auc)

    mlflow.log_artifact(FEATURE_ORDER_PATH)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        input_example=X_test.iloc[:5],
        registered_model_name=MODEL_NAME
    )

    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        mlflow.set_tag("git_commit", commit_hash)
    except Exception:
        mlflow.set_tag("git_commit", "unknown")

    print(f"\nModel logged to MLflow ({EXPERIMENT_NAME})")
    print(f"Accuracy: {acc:.3f} | AUC: {auc:.3f}")
