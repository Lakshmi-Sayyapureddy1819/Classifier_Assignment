# Notebook: notebook.ipynb

# ## 1. Imports and Setup
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import tensorflow as tf
from transformers import (
    DistilBertTokenizerFast,
    TFDistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ## 2. Load & Preprocess Data
df = pd.read_csv('data/reply_classification_dataset.csv')
df['label'] = df['label'].str.lower().map({
    'positive':'positive','posi­tive':'positive',
    'negative':'negative','ne­gative':'negative',
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

# ## 3. Baseline: Logistic Regression
vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tr_vec = vect.fit_transform(X_train)
X_te_vec = vect.transform(X_test)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_tr_vec, y_train)
y_pred_lr = lr.predict(X_te_vec)
print("LR Accuracy:", accuracy_score(y_test, y_pred_lr))
print("LR F1 (macro):", f1_score(y_test, y_pred_lr, average='macro'))
print(classification_report(y_test, y_pred_lr))

# ## 4. Prepare Transformer Dataset
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_enc = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
test_enc  = tokenizer(list(X_test),  truncation=True, padding=True, max_length=128)
label2id = {'negative':0,'neutral':1,'positive':2}
train_labels = [label2id[l] for l in y_train]
test_labels  = [label2id[l] for l in y_test]

class ReplyDataset(tf.data.Dataset):
    def __new__(cls, encodings, labels):
        return tf.data.Dataset.from_tensor_slices((dict(encodings), labels))

train_ds = ReplyDataset(train_enc, train_labels)
test_ds  = ReplyDataset(test_enc,  test_labels)

# ## 5. Fine-Tune distilBERT
model = TFDistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=3
)
training_args = TrainingArguments(
    output_dir='./model_output',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy='epoch',
    logging_dir='./logs'
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)
trainer.train()
trainer.evaluate()

# ## 6. Save Models & Artifacts
vect_filepath = 'model/vectorizer.pkl'
import joblib
joblib.dump(vect, vect_filepath)
model.save_pretrained('model/distilbert')
tokenizer.save_pretrained('model/distilbert')
