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

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float

app = FastAPI(title="Reply Classifier")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Tokenize input
    enc = tokenizer(req.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Get prediction
    with torch.no_grad():
        outputs = transformer_model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
    
    idx = int(torch.argmax(logits, dim=1))
    labels = ['negative','neutral','positive']
    
    return PredictResponse(label=labels[idx], confidence=probs[idx])

@app.get("/")
def root():
    return {"message": "Reply Classifier API"}
