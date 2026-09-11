# Clinical Data Pipeline & Predictive Analytics

An interactive, recruiter-facing engineering demonstration that takes clinical indicators through validation, preprocessing, predictive modeling, patient segmentation, and explainable API delivery.

> **Educational demonstration:** The application uses a reproducible synthetic cohort. It contains no real patient records or protected health information (PHI). Model outputs are not diagnoses or medical advice.

## What the demo shows

- Schema-validated clinical feature ingestion with FastAPI and Pydantic
- Median imputation and standardized preprocessing in an sklearn pipeline
- Logistic regression risk scoring with probability output
- K-Means patient-profile segmentation
- Model evaluation metrics including accuracy, precision, recall, and ROC–AUC
- Interactive, responsive browser experience with preconfigured sample profiles
- Health monitoring, automated tests, Docker packaging, and Render deployment

## Architecture

| Layer | Responsibility | Technology |
|---|---|---|
| Experience | Responsive analytics dashboard | HTML, CSS, JavaScript |
| API | Validation, prediction, pipeline metrics | FastAPI, Pydantic |
| ML pipeline | Imputation, scaling, classification | scikit-learn |
| Segmentation | Patient-profile grouping | K-Means |
| Operations | Packaging, health checks, CI | Docker, GitHub Actions, Render |

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Service and model readiness |
| `GET` | `/api/pipeline` | Pipeline stages and evaluation metrics |
| `POST` | `/api/predict` | Validate indicators and return model output |
| `GET` | `/docs` | Interactive OpenAPI documentation |

Example request:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age":54,"resting_bp":130,"cholesterol":245,"max_heart_rate":150,"exercise_angina":0,"st_depression":1.0}'
```

## Run locally

### Docker

```bash
docker build -t clinical-analytics-demo .
docker run --rm -p 8000:8000 clinical-analytics-demo
```

Open <http://localhost:8000>.

### Python

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Test

```bash
pytest -q
```

## Responsible-use notes

- Training records are generated deterministically for demonstration and are not a clinical study dataset.
- Metrics describe only the synthetic holdout set and must not be interpreted as clinical performance.
- Inputs and predictions are processed in memory and are not stored.
- A production clinical system would require governed data, bias analysis, calibration, external validation, security controls, regulatory review, and clinician oversight.

## Technology

Python 3.12 · FastAPI · Pydantic · NumPy · scikit-learn · pytest · Docker · GitHub Actions · Render
