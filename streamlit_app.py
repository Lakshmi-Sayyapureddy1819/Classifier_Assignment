import streamlit as st
import joblib
import torch
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

# ——— Load Artifacts ———
@st.cache_resource
def load_models():
    # Vectorizer & LR
    vect = joblib.load('model/vectorizer.pkl')
    lr = joblib.load('model/logistic_regression.pkl')
    # Transformer
    tokenizer = DistilBertTokenizerFast.from_pretrained('NirvanaLohitha/my-classifier-model')
    tf_model = DistilBertForSequenceClassification.from_pretrained('NirvanaLohitha/my-classifier-model')
    tf_model.eval()
    return vect, lr, tokenizer, tf_model

vect, lr_model, tokenizer, tf_model = load_models()
labels = ['negative', 'neutral', 'positive']

# ——— Streamlit UI ———
st.title("Reply Classification Demo")
st.write("Enter an email reply and get predictions from both models.")

text_input = st.text_area("Reply text", height=150)

if st.button("Classify"):
    if not text_input.strip():
        st.error("Please enter some text to classify.")
    else:
        # Logistic Regression inference
        X_vec = vect.transform([text_input])
        lr_label = lr_model.predict(X_vec)[0]
        lr_conf = lr_model.predict_proba(X_vec).max()

        # Transformer inference
        enc = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = tf_model(**enc).logits
            probs = torch.softmax(logits, dim=1).squeeze().tolist()
        idx = int(torch.argmax(logits, dim=1))
        tf_label = labels[idx]
        tf_conf = probs[idx]

        st.subheader("Results")
        st.write(f"**Logistic Regression:** {lr_label} (confidence: {lr_conf:.2f})")
        st.write(f"**DistilBERT Transformer:** {tf_label} (confidence: {tf_conf:.2f})")


