import os
import subprocess
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.exceptions import RestException

# === Load Environment Variables ===
load_dotenv()

REQUIRED_ENV_VARS = [
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD"
]

for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise EnvironmentError(f"❌ {var} is not set in .env")

# === Set MLflow Auth for DagsHub ===
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")
MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "pricing-ml-model")

# Auto-disable model registration on DagsHub
REGISTER_MODEL = False
if "dagshub.com" not in TRACKING_URI:
    REGISTER_MODEL = os.getenv("REGISTER_MODEL", "true").lower() == "true"
else:
    if os.getenv("REGISTER_MODEL", "").lower() == "true":
        print("⚠️  Model registration disabled due to DagsHub registry incompatibility.")

# === Set MLflow Tracking URI ===
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# === Paths ===
DATA_PATH = "data/raw/interactions.csv"
FEATURE_ORDER_PATH = "src/model/feature_order.pkl"
LOCAL_MODEL_PATH = "src/model/latest_model.joblib"
os.makedirs(os.path.dirname(FEATURE_ORDER_PATH), exist_ok=True)

# === Load and Preprocess Data ===
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["interaction_id", "customer_id", "product_id", "converted"])
X_encoded = pd.get_dummies(X, columns=["category", "segment", "time_of_day", "day_of_week", "season"])
y = df["converted"]

# Save feature order
joblib.dump(X_encoded.columns.tolist(), FEATURE_ORDER_PATH)

# Split → Train / Validation / Test
X_temp, X_test, y_temp, y_test = train_test_split(X_encoded, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

# === Build and Train Pipeline ===
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=42))
])
pipeline.fit(X_train, y_train)

# === Evaluation ===
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# Save local model (optional for local use)
joblib.dump(pipeline, LOCAL_MODEL_PATH)

# === MLflow Logging ===
with mlflow.start_run():
    # Parameters
    mlflow.log_params({
        "model": "logistic_regression",
        "scaling": True,
        "max_iter": 1000,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test)
    })

    # Metrics
    mlflow.log_metrics({
        "accuracy": acc,
        "auc": auc
    })

    # Git commit
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        commit_hash = "unknown"

    mlflow.set_tags({
        "git_commit": commit_hash,
        "stage": "development",
        "model_type": "logreg",
        "data_version": "v1"
    })

    # Artifacts
    mlflow.log_artifact(FEATURE_ORDER_PATH)

    # Log model
    try:
        if REGISTER_MODEL:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                input_example=X_test.iloc[:5]
            )
            print(f"📦 Model registered as: {MODEL_NAME}")
        else:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                input_example=X_test.iloc[:5]
            )
            print(f"📦 Model logged without registry (REGISTER_MODEL={REGISTER_MODEL})")

    except RestException as e:
        print("❌ Failed to register model due to unsupported registry.")
        print("💡 Use REGISTER_MODEL=false in .env or avoid using DagsHub for registry.")
        print(f"Error: {e}")

    # Final Output
    print("\n✅ Model logged successfully to MLflow.")
    print(f"🎯 Accuracy: {acc:.3f} | AUC: {auc:.3f}")
