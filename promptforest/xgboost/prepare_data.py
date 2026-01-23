import pandas as pd
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from collections import Counter
import math
import string
import os

def _load_dataset(dataset_name):
    print(f"[TASK] Loading {dataset_name}...")
    try:
        
        dataset_dict = load_dataset(dataset_name)
        for split, ds in dataset_dict.items():
            print(f"  - {split}: {len(ds)} samples")

        
        all_splits = [dataset_dict[split] for split in dataset_dict.keys()]
        combined_ds = concatenate_datasets(all_splits)
        df = combined_ds.to_pandas()
        
        cols = df.columns
        text_col = next((c for c in ['text', 'instruction', 'prompt', 'inputs'] if c in cols), None)
        label_col = next((c for c in ['label', 'labels'] if c in cols), None)
        
        if text_col and label_col:
            df = df.rename(columns={text_col: 'text', label_col: 'label'})
            return df[['text', 'label']]
        else:
            print(f"[WARN] Skipping {dataset_name}: Could not find text/label columns. Found: {cols}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"[ERR] Error loading {dataset_name}: {e}")
        return pd.DataFrame()

def main():
    DATASETS = [
        #@todo: Add more datasets
        "JasperLS/prompt-injections",
        "geekyrakshit/prompt-injection-dataset"
    ]
    
    dfs = [_load_dataset(name) for name in DATASETS]
    
    dfs = [df for df in dfs if not df.empty]
    
    if not dfs:
        print("[WARN] No data loaded from any dataset.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    
    # clean
    full_df.dropna(subset=['text', 'label'], inplace=True)
    full_df['text'] = full_df['text'].astype(str)
    full_df = full_df[full_df['text'].str.strip().str.len() > 0]
    
    print(f"[INFO] Total Combined Samples: {len(full_df)}")
    
    print("[TASK] Extracting heuristic features...")
    
    print("[TASK] Generating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(full_df['text'].tolist(), show_progress_bar=True, batch_size=64, device='mps')
    
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
