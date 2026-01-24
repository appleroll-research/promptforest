"""
Ensemble Inference Library for PromptForest.
"""

import os
import sys
import time
import joblib
import torch
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as transformers_logging
from sentence_transformers import SentenceTransformer
from .config import MODELS_DIR, XGB_MODEL_PATH, load_config

# Suppress Warnings
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Prevent deadlocks/warnings

MALICIOUS_KEYWORDS = ['unsafe', 'malicious', 'injection', 'attack', 'jailbreak']

def get_device(device_setting):
    if device_setting == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    return device_setting

class ModelInference:
    def predict(self, prompt):
        raise NotImplementedError

class HFModel(ModelInference):
    def __init__(self, name, dirname, settings):
        self.name = name
        self.path = MODELS_DIR / dirname
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self.malicious_idx = 1
        
        self.device_name = get_device(settings.get('device', 'auto'))
        self.fp16 = settings.get('fp16', True)
        
        self._load()

    def _load(self):
        if not self.path.exists():
             print(f"[WARN] Model path not found: {self.path}")
             return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.path)
            
            self.model.to(self.device_name)
            if self.device_name in ['cuda', 'mps'] and self.fp16:
                print("Using FP16 precision for model " + self.model.config._name_or_path)
                self.model.half()
            self.device = self.device_name
                
            self.model.eval()
            self._determine_label_map()
        except Exception as e:
            print(f"[ERR] Failed to load {self.name}: {e}")
            self.model = None

    def _determine_label_map(self):
        id2label = self.model.config.id2label
        found = False
        for idx, label in id2label.items():
            if any(kw in label.lower() for kw in MALICIOUS_KEYWORDS):
                self.malicious_idx = idx
                found = True
                break
        if not found:
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
            
            if self.malicious_idx < len(probs):
                return float(probs[self.malicious_idx])
            else:
                return float(probs[-1])
                
        except Exception:
            return 0.0


class XGBoostModel(ModelInference):
    def __init__(self, settings):
        self.name = "xgboost_custom"
        self.settings = settings
        self.model = None
        self.embedder = None
        
        self.device_name = get_device(settings.get('device', 'auto'))
        self.fp16 = settings.get('fp16', True)
        
        self._load()

    def _load(self):
        try:
            if os.path.exists(XGB_MODEL_PATH):
                self.model = joblib.load(XGB_MODEL_PATH)
            
            ST_PATH = MODELS_DIR / 'sentence_transformer'
            if ST_PATH.exists():
                self.embedder = SentenceTransformer(str(ST_PATH))
            else:
                print("Cannot find local SentenceTransformer model. Downloading...")
                # Fallback to download default if local cache missing
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            if self.device_name in ['cuda', 'mps']:
                self.embedder.to(self.device_name)
                if self.fp16:
                   try:
                       self.embedder[0].auto_model.half() 
                   except:
                       pass
        except Exception as e:
            print(f"[ERR] Failed to load XGBoost: {e}")
            self.model = None

    def predict(self, prompt):
        if not self.model or not self.embedder:
            return 0.0
        try:
            emb = self.embedder.encode([prompt])
            prob = self.model.predict_proba(emb)[0][1]
            return float(prob)
        except Exception:
            return 0.0


class EnsembleGuard:
    def __init__(self, config=None):
        """
        Initialize the EnsembleGuard.
        :param config: Dictionary containing configuration. If None, loads default/user config.
        """
        # Check if models need to be downloaded
        self._ensure_models_available()
        
        if config is None:
            self.config = load_config()
        else:
            self.config = config
            
        self.models = []
        self._init_models()
        self.device_used = get_device(self.config['settings'].get('device', 'auto'))
    
    def _ensure_models_available(self):
        """Check if models are available, download if needed."""
        from .config import MODELS_DIR
        
        # Check if models directory exists and has content
        if MODELS_DIR.exists() and any(MODELS_DIR.iterdir()):
            return
        
        # Models not found, download them
        print("Models not found. Downloading...")
        from .download import download_all
        download_all()

    def _init_models(self):
        settings = self.config.get('settings', {})
        model_configs = self.config.get('models', [])
        
        for model_cfg in model_configs:
            if not model_cfg.get('enabled', True):
                continue
                
            model_type = model_cfg.get('type')
            
            if model_type == 'hf':
                self.models.append(HFModel(model_cfg['name'], model_cfg['path'], settings))
            elif model_type == 'xgboost':
                self.models.append(XGBoostModel(settings))
            else:
                print(f"Unknown model type: {model_type}")

    def check_prompt(self, prompt):
        start_time = time.perf_counter()
        results = {}
        
        with ThreadPoolExecutor() as executor:
            future_to_model = {executor.submit(model.predict, prompt): model for model in self.models}
            for future in future_to_model:
                model = future_to_model[future]
                try:
                    prob = future.result()
                    results[model.name] = prob
                except Exception:
                    results[model.name] = 0.0

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        probs = list(results.values())
        if not probs:
             return {"error": "No models loaded"}

        avg_prob = np.mean(probs)
        max_prob = np.max(probs)
        
        # Uncertainty
        std_dev = np.std(probs)
        uncertainty_score = min(std_dev * 2, 1.0)
        
        is_malicious = avg_prob > 0.5
        
        response = {
            "is_malicious": bool(is_malicious),
            "confidence": float(avg_prob if is_malicious else 1 - avg_prob), 
            "uncertainty": float(uncertainty_score),
            "malicious_score": float(avg_prob),
            "max_risk_score": float(max_prob)
        }
        
        # Add stats if requested
        if self.config.get('logging', {}).get('stats', True):
            response["details"] = results
            response["latency_ms"] = duration_ms
            
        return response
