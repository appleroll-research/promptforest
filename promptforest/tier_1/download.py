"""
Script to download and save ensemble models locally.
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

MODELS = {
    "llama_guard": "meta-llama/Llama-Prompt-Guard-2-86M",
    "protectai_deberta": "protectai/deberta-v3-base-prompt-injection",
    "testsavantai_defender": "testsavantai/prompt-injection-defender-medium-v0",
    "deepset_deberta": "deepset/deberta-v3-base-injection",
    "katanemo_arch": "katanemo/Arch-Guard"
}

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models_ensemble')

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _download_hf_model(name, model_id):
    """Download and save a Hugging Face model and tokenizer."""
    print(f"[TASK] Downloading {name} ({model_id})...")
    save_path = os.path.join(MODELS_DIR, name)
    
    try:
        if os.path.exists(save_path):
            print(f"  - [INFO] Model already exists at {save_path}. Skipping.")
            return

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        print(f"  - [INFO] Labels: {model.config.id2label}")
        print(f"  - [INFO] Saved to {save_path}")
        
    except Exception as e:
        print(f"  - [ERR] Failed to download {name}: {e}")

def _download_sentence_transformer():
    """Download and save the SentenceTransformer model."""
    print(f"[TASK] Downloading SentenceTransformer ({EMBEDDING_MODEL_NAME})...")
    save_path = os.path.join(MODELS_DIR, 'sentence_transformer')
    
    try:
        # SentenceTransformer saves differently, usually we just let it cache or save_pretrained
        # We will save it explicitly to be consistent with "local" requirement
        if os.path.exists(save_path):
             print(f"  - [INFO] Model already exists at {save_path}. Skipping.")
             return

        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        model.save(save_path)
        print(f"  - [INFO] Saved to {save_path}")
        
    except Exception as e:
        print(f"  - [ERR] Failed to download SentenceTransformer: {e}")

def main():
    print("[START] Downloading models for ensemble...")
    _ensure_dir(MODELS_DIR)
    
    # Download HF Classification Models
    for name, model_id in MODELS.items():
        _download_hf_model(name, model_id)
        
    # Download Embedding Model for XGBoost
    _download_sentence_transformer()
    
    print("[DONE] All models processed.")

if __name__ == "__main__":
    main()
