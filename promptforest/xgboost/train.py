import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
import joblib

def main():
    data_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    if not os.path.exists(data_path):
        print(f"[ERR] Data file not found at {data_path}. Run prepare_data.py first.")
        return

    print("[TASK] Loading data...")
    # Read CSV (might be large with embeddings)
    df = pd.read_csv(data_path)
    
    df.dropna(inplace=True)

    X = df.drop('label', axis=1)
    y = df['label']

    print(f"[TASK] Training with {len(df)} samples and {X.shape[1]} features.")

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp
    )

    # 'BOOST IT!!!
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        min_child_weight=10,
        reg_alpha=0.01,
        reg_lambda=1.5,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
        early_stopping_rounds=35,
    )


    print("[TASK] Training model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )


    print("[TASK] Evaluating model...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib_path = os.path.join(os.path.dirname(__file__), 'xgb_model.pkl')
    joblib.dump(model, joblib_path)
    print(f"[INFO] Pickle model saved to {joblib_path}")

if __name__ == "__main__":
    print("[START] (xgboost.train)")
    main()
    print("[DONE]")
