# Reply Classification Service

## Project Structure
- data/: raw datasets
- model/: saved vectorizer & transformer
- notebook.ipynb: data prep & training
- app.py: FastAPI service
- requirements.txt: Python deps
- Dockerfile: container config
- answers.md: short-answer responses

## Setup Locally
1. Clone repo and navigate in:
```
git clone <repo_url>
cd reply-classifier
```

2. Install requirements:

```
pip install -r requirements.txt
```

3. Ensure `model/vectorizer.pkl` and `model/distilbert/` exist.

## Run API
```
uvicorn app:app --reload --port 8000
```

## Predict Endpoint
- URL: `POST /predict`
- Body:
{ "text": "Looking forward to the demo!" }


- Response:
  ```
{ "label": "positive", "confidence": 0.87 }
```

## Docker
1. Build:
```
docker build -t reply-classifier .
```

2. Run:
```
docker run -p 8000:8000 reply-classifier
```

