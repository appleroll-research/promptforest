"""
Script to download and save ensemble models locally.
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from .config import MODELS_DIR

# Configuration
MODELS = {
    "llama_guard": "meta-llama/Llama-Prompt-Guard-2-86M",
    "protectai_deberta": "protectai/deberta-v3-base-prompt-injection",
    "testsavantai_defender": "testsavantai/prompt-injection-defender-base-v1",
    "deepset_deberta": "deepset/deberta-v3-base-injection",
    "katanemo_arch": "katanemo/Arch-Guard"
}

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _download_hf_model(name, model_id):
    """Download and save a Hugging Face model and tokenizer."""
    print(f"[TASK] Downloading {name} ({model_id})...")
    save_path = MODELS_DIR / name
    
    try:
        if save_path.exists():
            print(f"  - [INFO] Model already exists at {save_path}. Skipping.")
            return

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        # Print label mapping for verification
        print(f"  - [INFO] Labels: {model.config.id2label}")
        print(f"  - [INFO] Saved to {save_path}")
        
    except Exception as e:
        print(f"  - [ERR] Failed to download {name}: {e}")

def _download_sentence_transformer():
    """Download and save the SentenceTransformer model."""
    print(f"[TASK] Downloading SentenceTransformer ({EMBEDDING_MODEL_NAME})...")
    save_path = MODELS_DIR / 'sentence_transformer'
    
    try:
        if save_path.exists():
             print(f"  - [INFO] Model already exists at {save_path}. Skipping.")
             return

        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        model.save(str(save_path))
        print(f"  - [INFO] Saved to {save_path}")
        
    except Exception as e:
        print(f"  - [ERR] Failed to download SentenceTransformer: {e}")

def download_all():
    print(f"[START] Downloading models to {MODELS_DIR}...")
    _ensure_dir(MODELS_DIR)
    
    # Download HF Classification Models
    for name, model_id in MODELS.items():
        _download_hf_model(name, model_id)
        
    # Download Embedding Model for XGBoost
    _download_sentence_transformer()
    
    print("[DONE] All models processed.")

if __name__ == "__main__":
    download_all()
