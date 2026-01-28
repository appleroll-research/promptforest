"""
Script to download Llama Guard 2 86M from custom GitHub releases.
Downloads files in parallel for speed.
"""

import os
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .config import MODELS_DIR

BASE_URL = "https://github.com/appleroll-research/promptforest-model-ensemble/releases/download/v0.5.0-alpha"
FILES_TO_DOWNLOAD = [
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json"
]

def _download_file(url, save_path):
    """Download a single file."""
    if save_path.exists():
        return
        
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        # Clean up partial file
        if save_path.exists():
            os.remove(save_path)

def download_llama_guard():
    """Download Llama Guard files in parallel."""
    save_dir = MODELS_DIR / "llama_guard"
    
    # Check if all files exist
    if save_dir.exists() and all((save_dir / f).exists() for f in FILES_TO_DOWNLOAD):
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for filename in FILES_TO_DOWNLOAD:
            url = f"{BASE_URL}/{filename}"
            save_path = save_dir / filename
            futures.append(executor.submit(_download_file, url, save_path))
            
        for future in futures:
            future.result()
            
