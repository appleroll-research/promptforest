import pickle
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---- Load your XGBoost model ----
with open("xgb_model_new.pkl", "rb") as f:
    model = pickle.load(f)

# ---- Load embedder ----
device = "mps" if torch.backends.mps.is_available() else "cpu"
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# ---- Load Qualifire benchmark ----
# This will load the dataset from Hugging Face. It has "text" and "label" fields
dataset = load_dataset("qualifire/prompt-injections-benchmark", split="test")

# Map string labels to {0,1}
def map_label(example):
    example["label_int"] = 1 if example["label"].lower() == "jailbreak" else 0
    return example

dataset = dataset.map(map_label)

texts = dataset["text"]
labels = dataset["label_int"]

# ---- Encode embeddings ----
print("Generating embeddings...")
embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64, device=device)

# ---- Predict ----
print("Running inference...")
preds = model.predict(embeddings)

# ---- Metrics ----
acc = accuracy_score(labels, preds)
prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

print("===== Benchmark Results =====")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
