# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get Hugging Face token from .env
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    raise ValueError("❌ Hugging Face token not found. Please set HF_TOKEN in .env")


# Load artifacts
vectorizer = joblib.load('model/vectorizer.pkl')
tokenizer = DistilBertTokenizerFast.from_pretrained('NirvanaLohitha/my-classifier-model')
transformer_model = DistilBertForSequenceClassification.from_pretrained('NirvanaLohitha/my-classifier-model')
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
