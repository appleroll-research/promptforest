import pandas as pd
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import torch
import sys

def _load_dataset(dataset_name):
    print(f"[TASK] Loading {dataset_name}...")
    try:
        if dataset_name == "allenai/wildjailbreak":
            # allenAI handles stuff differently
            # their train dataset is in 'eval' subset, 'train' split
            dataset_dict = load_dataset(dataset_name, 'eval')
        else:
            dataset_dict = load_dataset(dataset_name)
        # gather all splits
        splits = []
        if hasattr(dataset_dict, 'values'): # DatasetDict
            splits.extend(dataset_dict.values())
        else:
            # Maybe it's a Dataset?
            splits.append(dataset_dict)
            
        if not splits:
            return pd.DataFrame()

        print(f"  - Found {len(splits)} splits.")
        combined_ds = concatenate_datasets(splits)
        df = combined_ds.to_pandas()
        
        cols = df.columns
        # Flexible column mapping
        text_col = next((c for c in ['text', 'instruction', 'prompt', 'inputs', 'input', 'adversarial'] if c in cols), None)
        label_col = next((c for c in ['label', 'labels', 'class', 'classification', 'type'] if c in cols), None)
        
        if text_col and label_col:
            df = df.rename(columns={text_col: 'text', label_col: 'label'})
            return df[['text', 'label']]
        else:
            print(f"[WARN] Skipping {dataset_name}: Could not find text/label columns. Found: {cols}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"[ERR] Error loading {dataset_name}: {e}")
        return pd.DataFrame()

def normalize_label(val):
    """Normalize labels to 0 (benign) and 1 (malicious)."""
    if isinstance(val, (int, float, np.integer, np.floating)):
        return int(val)
    
    s = str(val).lower().strip()
    # Common malicious labels
    if s in ['malicious', 'jailbreak', 'injection', 'unsafe', '1', 'bad', 'harmful']:
        return 1
    # Common benign labels
    if s in ['benign', 'safe', 'normal', '0', 'good', 'harmless']:
        return 0
    
    # Try converting numeric string
    try:
        return int(float(s))
    except:
        pass
        
    return None # Mark for removal

def main():
    DATASETS = [
        "JasperLS/prompt-injections",
        "geekyrakshit/prompt-injection-dataset",
        "allenai/wildjailbreak",
        "hendzh/PromptShield",
        "DhruvTre/jailbreakbench-paraphrase-2025-08",
        "jackhhao/jailbreak-classification"
    ]
    
    dfs = [_load_dataset(name) for name in DATASETS]
    dfs = [df for df in dfs if not df.empty]
    
    if not dfs:
        print("[WARN] No data loaded from any dataset.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Clean
    full_df.dropna(subset=['text', 'label'], inplace=True)
    full_df['text'] = full_df['text'].astype(str)
    full_df = full_df[full_df['text'].str.strip().str.len() > 0]
    
    print(f"[INFO] Total Combined Samples (Raw): {len(full_df)}")
    
    # Normalize labels
    print("[TASK] Normalizing labels...")
    full_df['label'] = full_df['label'].apply(normalize_label)
    
    # Filter out invalid labels
    initial_len = len(full_df)
    full_df = full_df.dropna(subset=['label'])
    full_df['label'] = full_df['label'].astype(int)
    print(f"[INFO] Removed {initial_len - len(full_df)} samples with invalid labels. Total: {len(full_df)}")
    
    print("[TASK] Generating embeddings...")
    # Determine device
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    print(f"[INFO] Using device: {device}")
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    embeddings = model.encode(full_df['text'].tolist(), show_progress_bar=True, batch_size=64, device=device)
    
    emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=full_df.index)
    
    output_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    
    final_df = pd.concat([full_df[['label']], emb_df], axis=1)
    
    if final_df.empty:
        print("[WARN] Final DataFrame is empty. Nothing to save.")
        return

    final_df.to_csv(output_path, index=False)
    print(f"[INFO] Processed data saved to {output_path} with {final_df.shape[1]} columns.")

if __name__ == "__main__":
    print("[START] (xgboost.prepare_data)")
    main()
    print("[DONE]")
