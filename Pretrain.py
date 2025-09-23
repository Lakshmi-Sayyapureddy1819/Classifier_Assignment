# ## 1. Imports and Setup
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import joblib


# ## 2. Load & Preprocess Data
df = pd.read_csv('data/reply_classification_dataset.csv')
df['label'] = df['label'].str.lower().map({
    'positive':'positive',
    'negative':'negative',
    'neutral':'neutral'
}).fillna('neutral')


def clean_text(text):
    t = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text).lower())
    return re.sub(r'\s+', ' ', t).strip()


df['clean_reply'] = df['reply'].apply(clean_text)
X = df['clean_reply']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset loaded: {len(df)} samples")
print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
print(f"Label distribution: {y.value_counts().to_dict()}")


# ## 3. Baseline: Logistic Regression
print("\n" + "="*50)
print("TRAINING LOGISTIC REGRESSION BASELINE")
print("="*50)

vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tr_vec = vect.fit_transform(X_train)
X_te_vec = vect.transform(X_test)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_tr_vec, y_train)
y_pred_lr = lr.predict(X_te_vec)

print("LR Accuracy:", accuracy_score(y_test, y_pred_lr))
print("LR F1 (macro):", f1_score(y_test, y_pred_lr, average='macro'))
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_lr))


# ## 4. Prepare Transformer Dataset
print("\n" + "="*50)
print("PREPARING TRANSFORMER DATA")
print("="*50)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_enc = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
test_enc = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

label2id = {'negative':0,'neutral':1,'positive':2}
train_labels = [label2id[l] for l in y_train]
test_labels = [label2id[l] for l in y_test]


class ReplyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_ds = ReplyDataset(train_enc, train_labels)
test_ds = ReplyDataset(test_enc, test_labels)

print("Tokenization completed")
print(f"Training samples: {len(train_ds)}")
print(f"Test samples: {len(test_ds)}")


# ## 5. Fine-Tune distilBERT (Manual Training Loop)
print("\n" + "="*50)
print("TRAINING TRANSFORMER MODEL")
print("="*50)

# Create model directory
os.makedirs('model', exist_ok=True)

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=3
)

# Create data loaders
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

print(f"Training on device: {device}")

# Training loop
model.train()
for epoch in range(3):
    total_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if i % 50 == 0:
            print(f'Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}')
    
    print(f'Epoch {epoch+1}/3 completed, Average Loss: {total_loss/len(train_loader):.4f}')

# Evaluate
print("\nEvaluating transformer model...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted = torch.argmax(outputs.logits, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

transformer_accuracy = correct/total
print(f'Transformer Accuracy: {transformer_accuracy:.4f}')


# ## 6. Save Models & Artifacts
print("\n" + "="*50)
print("SAVING MODELS")
print("="*50)

# Save vectorizer
joblib.dump(vect, 'model/vectorizer.pkl')
print("✓ Vectorizer saved to model/vectorizer.pkl")

# Save logistic regression model
joblib.dump(lr, 'model/logistic_regression.pkl')
print("✓ Logistic Regression saved to model/logistic_regression.pkl")

# Save transformer model
model.save_pretrained('model/distilbert')
tokenizer.save_pretrained('model/distilbert')
print("✓ Transformer model saved to model/distilbert/")

print("\n" + "="*50)
print("TRAINING COMPLETE - SUMMARY")
print("="*50)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Transformer Accuracy: {transformer_accuracy:.4f}")
print("\nAll models saved successfully!")
print("You can now run the FastAPI server with: uvicorn app:app --reload")