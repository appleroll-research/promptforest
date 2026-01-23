"""
Ensemble Inference Script for Tier 1 Prompt Guard.
Combines multiple HF models and a custom XGBoost model for robust detection.
"""

import os
import sys
import time
import joblib
import torch
import numpy as np
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as transformers_logging
from sentence_transformers import SentenceTransformer

# Suppress Warnings
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Prevent deadlocks/warnings

# Configuration
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models_ensemble')
XGB_MODEL_PATH = os.path.join(BASE_DIR, 'xgboost', 'xgb_model.pkl')

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Optimization Configuration
FORCE_FP32 = False        # Set to True to force FP32 precision (even on GPU/MPS)
QUANTIZE_8BIT_CPU = True # Set to True to use 8-bit dynamic quantization (Only if running on CPU)

# Derived Settings
# Use FP16 if we are on a GPU device and user hasn't forced FP32
USE_FP16 = (DEVICE in ['cuda', 'mps']) and (not FORCE_FP32)

# Verify logic for user
if DEVICE == 'cpu':
    print(f"[INFO] Running on CPU. Quantization: {'ENABLED (8-bit)' if QUANTIZE_8BIT_CPU else 'DISABLED (FP32)'}")
else:
    print(f"[INFO] Running on {DEVICE.upper()}. Precision: {'FP16' if USE_FP16 else 'FP32'}")

# Model Definitions (Must match download.py)
HF_MODELS_CONFIG = {
    "llama_guard": "llama_guard", 
    "protectai": "protectai_deberta",
    "testsavantai": "testsavantai_defender",
    "deepset": "deepset_deberta",
    "katanemo": "katanemo_arch"
}

# Keywords to identify the 'Malicious' class in label strings
MALICIOUS_KEYWORDS = ['unsafe', 'malicious', 'injection', 'attack', 'jailbreak']

class ModelInference:
    """Base class for model inference."""
    def predict(self, prompt):
        raise NotImplementedError

class HFModel(ModelInference):
    """Wrapper for Hugging Face Sequence Classification models."""
    def __init__(self, name, relative_path):
        self.name = name
        self.path = os.path.join(MODELS_DIR, relative_path)
        self.model = None
        self.tokenizer = None
        self.malicious_idx = 1 # Default assumption
        self._load()

    def _load(self):
        # print(f"[TASK] Loading {self.name}...")
        if not os.path.exists(self.path):
             print(f"[WARN] Model directory not found for {self.name} at {self.path}. Skipping.")
             self.model = None
             return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.path)
            
            # Optimization Logic
            if DEVICE == 'cpu' and QUANTIZE_8BIT_CPU:
                # 8-bit Dynamic Quantization (Only applies if actually running on CPU)
                # print(f"  - [{self.name}] Optimizing: 8-bit Quantization (CPU)")
                self.model.to('cpu')
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                self.device = 'cpu'
            else:
                self.model.to(DEVICE)
                if USE_FP16:
                    # print(f"  - [{self.name}] Optimizing: FP16 ({DEVICE})")
                    self.model.half() # Convert to Float16
                self.device = DEVICE
                
            self.model.eval()
            self._determine_label_map()
        except Exception as e:
            print(f"[ERR] Failed to load {self.name}: {e}")
            self.model = None

    def _determine_label_map(self):
        """Heuristic to determine which label index is 'Malicious'."""
        id2label = self.model.config.id2label
        # print(f"  - [{self.name}] Labels: {id2label}")
        
        # Look for malicious keywords
        found = False
        for idx, label in id2label.items():
            if any(kw in label.lower() for kw in MALICIOUS_KEYWORDS):
                self.malicious_idx = idx
                found = True
                break
        
        # If not found, assume 1 is malicious if binary (0, 1)
        if not found:
            print(f"  - [{self.name}] Warning: Could not detect malicious label, assuming index 1.")
            self.malicious_idx = 1 

    def predict(self, prompt):
        if not self.model:
            return 0.0
            
        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            probs = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()[0]
            
            # Return probability of being malicious
            if self.malicious_idx < len(probs):
                return float(probs[self.malicious_idx])
            else:
                return float(probs[-1]) # Fallback
                
        except Exception as e:
            print(f"[ERR] Error in {self.name} inference: {e}")
            return 0.0


class XGBoostModel(ModelInference):
    """Wrapper for the custom XGBoost model."""
    def __init__(self):
        self.name = "xgboost_custom"
        self.model = None
        self.embedder = None
        self._load()

    def _load(self):
        print(f"[TASK] Loading {self.name}...")
        try:
            # Load XGBoost
            if os.path.exists(XGB_MODEL_PATH):
                self.model = joblib.load(XGB_MODEL_PATH)
            else:
                print(f"[ERR] XGBoost model not found at {XGB_MODEL_PATH}")
                
            # Load Embedder
            ST_PATH = os.path.join(MODELS_DIR, 'sentence_transformer')
            if os.path.exists(ST_PATH):
                self.embedder = SentenceTransformer(ST_PATH)
            else:
                print(f"[WARN] Local SentenceTransformer not found, downloading default...")
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Optimization for Embedder
            if DEVICE in ['cuda', 'mps']:
                self.embedder.to(DEVICE)
                if USE_FP16:
                   # Try to cast to FP16 if supported
                   try:
                       self.embedder[0].auto_model.half() 
                   except:
                       pass # Fallback if structure is different
            elif DEVICE == 'cpu' and QUANTIZE_8BIT_CPU:
                pass # SentenceTransformer quantization is complex, skipping to ensure stability
                
        except Exception as e:
            print(f"[ERR] Failed to load XGBoost/Embedder: {e}")
            self.model = None

    def predict(self, prompt):
        if not self.model or not self.embedder:
            return 0.0
            
        try:
            # Embed
            emb = self.embedder.encode([prompt])
            # Create DataFrame with correct column names as XGBoost expects
            # However, simpler to just pass numpy array if model allows, but features might need matching
            # train.py saved it using a dataframe with columns emb_0...emb_383
            # XGBoost might warn about feature names but should work with numpy array if order is preserved
            
            prob = self.model.predict_proba(emb)[0][1] # Probability of class 1 (Assumed Malicious)
            return float(prob)
        except Exception as e:
            print(f"[ERR] Error in XGBoost inference: {e}")
            return 0.0


class EnsembleGuard:
    def __init__(self):
        self.models = []
        self._init_models()

    def _init_models(self):
        # Initialize HF Models
        for name, path in HF_MODELS_CONFIG.items():
            self.models.append(HFModel(name, path))
            
        # Initialize XGBoost
        self.models.append(XGBoostModel())

    def check_prompt(self, prompt):
        """
        Run all models in parallel and return ensemble result.
        """
        start_time = time.perf_counter()
        
        results = {}
        
        # Parallel Execution
        with ThreadPoolExecutor() as executor:
            future_to_model = {executor.submit(model.predict, prompt): model for model in self.models}
            
            for future in future_to_model:
                model = future_to_model[future]
                try:
                    prob = future.result()
                    results[model.name] = prob
                except Exception as e:
                    print(f"[ERR] {model.name} failed: {e}")
                    results[model.name] = 0.0 # Fail safe

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Voting Logic
        probs = list(results.values())
        
        # 1. Average Probability
        avg_prob = np.mean(probs)
        
        # 2. Max Probability
        max_prob = np.max(probs)
        
        # 3. Uncertainty Score (Discrepancy)
        # Based on Standard Deviation. Max std for bounded [0,1] is 0.5.
        # We normalize (std * 2) so 0 = full agreement, 1 = max disagreement.
        std_dev = np.std(probs)
        uncertainty_score = min(std_dev * 2, 1.0)
        
        # 4. Soft Voting decision (Threshold 0.5)
        is_malicious = avg_prob > 0.5
        
        return {
            "is_malicious": bool(is_malicious),
            "confidence": float(avg_prob if is_malicious else 1 - avg_prob), # Confidence in the decision
            "uncertainty": float(uncertainty_score),
            "malicious_score": float(avg_prob),
            "max_risk_score": float(max_prob),
            "details": results,
            "latency_ms": duration_ms
        }

def main(prompt):
    print(f"\n[INFO] Initializing Ensemble (Device: {DEVICE})...")
    ensemble = EnsembleGuard()
    
    print(f"\n[TASK] Analyzing Prompt: '{prompt}'")
    result = ensemble.check_prompt(prompt)
    
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {'MALICIOUS' if result['is_malicious'] else 'BENIGN'}")
    print(f"{'='*60}")
    print(f"Ensemble Score (Avg Risk): {result['malicious_score']*100:.2f}%")
    print(f"Max Risk Score:            {result['max_risk_score']*100:.2f}%")
    print(f"Uncertainty (Discrepancy): {result['uncertainty']*100:.2f}%")
    print(f"Confidence:                {result['confidence']*100:.2f}%")
    print(f"Latency:                   {result['latency_ms']:.2f} ms")
    print("-" * 60)
    print("Individual Model Scores (Probability of Malicious):")
    for name, score in result['details'].items():
        print(f"  - {name:<20}: {score*100:.2f}%")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ensemble_inf.py '<prompt>'")
        sys.exit(1)
        
    main(sys.argv[1])
