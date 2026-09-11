from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "UP"


def test_prediction_contract():
    response = client.post("/api/predict", json={
        "age": 54, "resting_bp": 130, "cholesterol": 245,
        "max_heart_rate": 150, "exercise_angina": 0, "st_depression": 1.0,
    })
    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["probability"] <= 1
    assert result["riskLevel"] in {"Lower", "Moderate", "Elevated"}


def test_rejects_out_of_range_input():
    response = client.post("/api/predict", json={
        "age": 12, "resting_bp": 130, "cholesterol": 245,
        "max_heart_rate": 150, "exercise_angina": 0, "st_depression": 1.0,
    })
    assert response.status_code == 422
