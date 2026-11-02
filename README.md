# Heart Disease Prediction API

A FastAPI service that predicts whether heart disease is present from 13 clinical features. It runs a scikit-learn classifier trained on the [Kaggle Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). The app is packaged with Docker and deployed on Render.

| | |
|---|---|
| **Live API** | https://heart-disease-api.onrender.com *(replace with your Render URL)* |
| **Swagger UI** | https://heart-disease-api.onrender.com/docs |

> The free Render plan puts the service to sleep when it's idle, so the first request can take up to about 50 seconds.

## Project structure

```
heart-disease-api/
├── app/
│   ├── main.py          # FastAPI app: /health, /info, /predict
│   └── schemas.py       # Pydantic input/output models
├── model/
│   ├── heart_model.joblib   # trained model (StandardScaler + LogisticRegression)
│   └── metadata.json        # model type, feature list, test accuracy
├── data/heart.csv       # dataset (1025 rows)
├── train.py             # trains the model and writes model/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Model

- **Data:** 1025 rows, 13 features, binary `target` (1 = heart disease, 0 = no heart disease)
- **Model:** `StandardScaler` followed by `LogisticRegression`, using an 80/20 stratified split
- **Test accuracy:** about 0.81. Accuracy wasn't the goal of this assignment.

To retrain, use the same Python version and packages as the Docker image:

```bash
docker run --rm -v "$PWD:/work" -w /work python:3.12-slim \
  sh -c "pip install -r requirements.txt && python train.py"
```

## Run locally with Docker

```bash
docker-compose build
docker-compose up
```

Then open http://localhost:8000/docs.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check: returns `{"status": "ok"}` |
| GET | `/info` | Model type, feature list, test accuracy |
| POST | `/predict` | Returns `heart_disease` (true/false) and the probability |

### Example requests

```bash
curl http://localhost:8000/health

curl http://localhost:8000/info

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":41,"sex":0,"cp":1,"trestbps":130,"chol":204,"fbs":0,"restecg":0,"thalach":172,"exang":0,"oldpeak":1.4,"slope":2,"ca":0,"thal":2}'
```

Response:

```json
{"heart_disease": true, "probability": 0.9622}
```

### Input features

| Feature | Meaning |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment (0–2) |
| ca | Number of major vessels colored by fluoroscopy (0–4) |
| thal | Thalassemia (0–3) |

## Deploy on Render

1. Push this repository to GitHub.
2. On Render, choose **New → Web Service** and connect the repository.
3. Set **Language** to **Docker**, leave **Root Directory** empty (the Dockerfile is at the repo root), and pick the **Free** instance type.
4. Click **Deploy**. Render sets `$PORT`, and the Dockerfile's `CMD` already uses it.
5. Test the live service with the `curl` commands above, replacing `http://localhost:8000` with the Render URL.
