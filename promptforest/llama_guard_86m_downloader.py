"""
Script to download Llama Guard 2 86M from a custom GitHub repository.
Handles split safetensor files and combines them locally.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from .config import MODELS_DIR

LLAMA_GUARD_REPO = "https://github.com/appleroll-research/promptforest-model-ensemble.git"

def _download_llama_guard():
    """Download Llama Guard from custom repository and combine split files."""
    save_path = MODELS_DIR / "llama_guard"
    
    if save_path.exists():
        return
    
    try:
        # Use temporary directory for cloning
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Clone repository silently
            subprocess.run(
                ["git", "clone", "--depth", "1", LLAMA_GUARD_REPO, str(temp_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            
            # Get the llama_guard folder from the cloned repo
            source_dir = temp_path / "llama_guard"
            if not source_dir.exists():
                raise FileNotFoundError(f"llama_guard folder not found in repository")
            
            # Copy to models directory
            save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir, save_path)
            
            # Combine split safetensor files
            model_file = save_path / "model.safetensors"
            if not model_file.exists():
                # Find and combine c_* files
                split_files = sorted(save_path.glob("c_*"))
                if split_files:
                    with open(model_file, 'wb') as outfile:
                        for split_file in split_files:
                            with open(split_file, 'rb') as infile:
                                outfile.write(infile.read())
                    
                    # Delete split files
                    for split_file in split_files:
                        split_file.unlink()
        
    except Exception as e:
        # Clean up on failure
        if save_path.exists():
            shutil.rmtree(save_path)
        raise Exception(f"Failed to download Llama Guard: {e}")

def download_llama_guard():
    """Public interface for downloading Llama Guard."""
    _download_llama_guard()
