from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "resting_bp", "cholesterol", "max_heart_rate", "exercise_angina", "st_depression"]
STATIC = Path(__file__).parent / "static"


class PatientInput(BaseModel):
    age: int = Field(54, ge=18, le=90)
    resting_bp: int = Field(130, ge=80, le=220)
    cholesterol: int = Field(245, ge=100, le=600)
    max_heart_rate: int = Field(150, ge=60, le=220)
    exercise_angina: Literal[0, 1] = 0
    st_depression: float = Field(1.0, ge=0, le=7)


def build_models():
    """Create a reproducible synthetic cohort and train demo models."""
    rng = np.random.default_rng(42)
    count = 1200
    age = np.clip(rng.normal(54, 10, count), 29, 82)
    bp = np.clip(rng.normal(132, 18, count), 90, 210)
    cholesterol = np.clip(rng.normal(242, 48, count), 120, 520)
    max_hr = np.clip(205 - age + rng.normal(0, 15, count), 70, 205)
    angina = rng.binomial(1, np.clip((age - 35) / 75, .08, .62))
    depression = np.clip(rng.gamma(1.3, .9, count) + angina * .55, 0, 6.5)
    X = np.column_stack([age, bp, cholesterol, max_hr, angina, depression])

    logits = (-7.1 + .047 * age + .012 * (bp - 120) + .006 * (cholesterol - 200)
              - .020 * (max_hr - 140) + 1.05 * angina + .65 * depression)
    probability = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probability)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("prepare", ColumnTransformer([("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), list(range(len(FEATURES))))])),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]
    metrics = {
        "recordsProcessed": count,
        "featuresValidated": len(FEATURES),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 3),
        "rocAuc": round(float(roc_auc_score(y_test, scores)), 3),
    }
    scaled = StandardScaler().fit_transform(X)
    clusterer = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaled)
    return model, clusterer, StandardScaler().fit(X), metrics


MODEL, CLUSTERER, CLUSTER_SCALER, METRICS = build_models()
app = FastAPI(title="Clinical Data Pipeline & Predictive Analytics", version="1.0.0")


@app.get("/api/health")
def health():
    return {"status": "UP", "model": "ready", "data": "synthetic"}


@app.get("/api/pipeline")
def pipeline():
    return {
        **METRICS,
        "stages": ["Ingest", "Validate", "Transform", "Train", "Evaluate", "Serve"],
        "dataNotice": "Synthetic educational cohort; no real patient records or PHI.",
    }


@app.post("/api/predict")
def predict(patient: PatientInput):
    values = np.array([[getattr(patient, feature) for feature in FEATURES]], dtype=float)
    probability = float(MODEL.predict_proba(values)[0, 1])
    cluster = int(CLUSTERER.predict(CLUSTER_SCALER.transform(values))[0])
    label = "Lower" if probability < .35 else "Moderate" if probability < .65 else "Elevated"
    segment = ["Active baseline", "Cardio-metabolic watch", "Elevated indicator profile"][cluster]
    factors = sorted([
        ("Age", max(0, (patient.age - 45) / 35)),
        ("Resting blood pressure", max(0, (patient.resting_bp - 120) / 80)),
        ("Cholesterol", max(0, (patient.cholesterol - 190) / 250)),
        ("Exercise-induced angina", float(patient.exercise_angina)),
        ("ST depression", patient.st_depression / 6),
        ("Maximum heart rate", max(0, (150 - patient.max_heart_rate) / 80)),
    ], key=lambda item: item[1], reverse=True)[:3]
    return {
        "riskLevel": label,
        "probability": round(probability, 3),
        "segment": segment,
        "cluster": cluster + 1,
        "topIndicators": [name for name, score in factors if score > 0],
        "disclaimer": "Educational model output only. Not a diagnosis or medical advice.",
    }


app.mount("/assets", StaticFiles(directory=STATIC), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str):
    return FileResponse(STATIC / "index.html")
