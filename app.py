# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load artifacts
vectorizer = joblib.load('model/vectorizer.pkl')
tokenizer = DistilBertTokenizerFast.from_pretrained('model/distilbert')
transformer_model = DistilBertForSequenceClassification.from_pretrained('model/distilbert')
transformer_model.eval()

# Request/Response schemas
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float

app = FastAPI(title="Reply Classifier")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Baseline vector features (optional fallback)
    vec = vectorizer.transform([req.text])
    # Transformer inference
    enc = tokenizer(req.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = transformer_model(**enc).logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
    idx = int(torch.argmax(logits, dim=1))
    labels = ['negative','neutral','positive']
    return PredictResponse(label=labels[idx], confidence=probs[idx])
