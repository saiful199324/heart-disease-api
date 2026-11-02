"""Train a heart disease classifier and save it to model/heart_model.joblib."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"

df = pd.read_csv(ROOT / "data" / "heart.csv")
X = df[FEATURES].to_numpy()
y = df["target"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODEL_DIR / "heart_model.joblib")
(MODEL_DIR / "metadata.json").write_text(json.dumps({
    "model_type": "LogisticRegression (StandardScaler pipeline)",
    "features": FEATURES,
    "test_accuracy": round(accuracy, 4),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
}, indent=2))

print(f"Saved model to {MODEL_DIR / 'heart_model.joblib'} (test accuracy {accuracy:.3f})")
