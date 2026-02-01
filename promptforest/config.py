import os
import yaml # type: ignore
from pathlib import Path

# Package-internal paths (for assets like the XGBoost classifier pickle)
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
XGB_MODEL_PATH = os.path.join(PACKAGE_DIR, 'xgboost', 'xgb_model.pkl')

# User data paths (for models)
USER_DATA_DIR = Path.home() / ".promptforest"
MODELS_DIR = USER_DATA_DIR / "models"

DEFAULT_CONFIG = {
    "models": [
        {"name": "llama_guard", "path": "llama_guard", "type": "hf", "enabled": True, "accuracy_weight": 1.0},
        {"name": "vijil", "path": "vijil_dome", "type": "hf", "enabled": True, "accuracy_weight": 1.0},
        {"name": "xgboost", "type": "xgboost", "enabled": True, "threshold": 0.10, "accuracy_weight": 1.0}
    ],
    "settings": {
        "device": "auto",  # Options: auto, cuda, mps, cpu
        "fp16": True       # Use FP16 precision on GPU/MPS (only applies if device is GPU/MPS)
    },
    "logging": {
        "stats": True
    }
}

def load_config(config_path=None):
    """
    Load configuration from a YAML file, merging with defaults.
    """
    # Start with a deep copy of the default config structure
    config = {
        "models": [m.copy() for m in DEFAULT_CONFIG["models"]],
        "settings": DEFAULT_CONFIG["settings"].copy(),
        "logging": DEFAULT_CONFIG["logging"].copy()
    }
    
    if config_path:
        path = Path(config_path)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    if user_config:
                        # 1. Merge Settings
                        if "settings" in user_config:
                            config["settings"].update(user_config["settings"])
                            
                        # 2. Merge Logging
                        if "logging" in user_config:
                            config["logging"].update(user_config["logging"])
                            
                        # 3. Merge Models (Smart Merge)
                        if "models" in user_config:
                            user_models = user_config["models"]
                            if isinstance(user_models, list):
                                # Convert current models to dict for easy lookup by name
                                existing_model_map = {m["name"]: m for m in config["models"]}
                                
                                for u_model in user_models:
                                    name = u_model.get("name")
                                    if name and name in existing_model_map:
                                        # Update existing model configuration (e.g. enable/disable, change path)
                                        existing_model_map[name].update(u_model)
                                    else:
                                        # Add new custom model
                                        config["models"].append(u_model)
                                        
            except Exception as e:
                print(f"Warning: Failed to load config from {path}: {e}")
        else:
            print(f"Warning: Config file {path} not found. Using defaults.")
            
    return config
