"""
Script to download and save ensemble models locally.
"""

import os
import sys
import threading
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from .config import MODELS_DIR
from .llama_guard_86m_downloader import download_llama_guard

# Configuration
MODELS = {
    "protectai": "protectai/deberta-v3-base-prompt-injection-v2",
    "vijil_dome": "vijil/vijil_dome_prompt_injection_detection"
}

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _download_hf_model(name, model_id):
    """Download and save a Hugging Face model and tokenizer."""
    save_path = MODELS_DIR / name
    
    try:
        if save_path.exists():
            return
        
        # Special handling for Vijil (ModernBERT tokenizer issue)
        tokenizer_id = model_id
        if "vijil" in name or "vijil" in model_id:
            tokenizer_id = "answerdotai/ModernBERT-base"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
    except Exception as e:
        print(f"Failed to download {model_id}: {e}")

def _download_sentence_transformer():
    """Download and save the SentenceTransformer model."""
    # print(f"Downloading SentenceTransformer ({EMBEDDING_MODEL_NAME})...")
    save_path = MODELS_DIR / 'sentence_transformer'
    
    try:
        if save_path.exists():
            #  print(f"  - Model already exists at {save_path}. Skipping.")
             return

        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        model.save(str(save_path))
        #print(f"  - Saved to {save_path}")
        
    except Exception as e:
        print(f"SentenceTransformer download failed: {e}")

def download_all():
    print(f"Downloading models to {MODELS_DIR}...")
    _ensure_dir(MODELS_DIR)
    
    # Download Llama Guard in parallel (slowest download)
    llama_guard_thread = threading.Thread(target=download_llama_guard, daemon=False)
    llama_guard_thread.start()
    
    # Download HF Classification Models
    for name, model_id in MODELS.items():
        _download_hf_model(name, model_id)
        
    # Download Embedding Model for XGBoost
    _download_sentence_transformer()
    
    # Wait for Llama Guard to complete
    llama_guard_thread.join()
    
    print("All models downloaded.")

if __name__ == "__main__":
    download_all()
