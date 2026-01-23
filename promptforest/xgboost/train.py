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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize XGBoost
    # Using tree_method='hist' is generally faster for larger datasets
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=150,           # Slightly increased
        learning_rate=0.1,
        max_depth=6,                # Slightly deeper for complex embeddings
        use_label_encoder=False,
        eval_metric='logloss'
    )

    print("[TASK] Training model...")
    model.fit(X_train, y_train)

    print("[TASK] Evaluating model...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature Importance (Top 20)
    print("\nTop 20 Feature Importances:")
    importances = model.feature_importances_
    features = X.columns
    
    # Create a DataFrame for sorting
    feat_imp = pd.DataFrame({'feature': features, 'importance': importances})
    feat_imp = feat_imp.sort_values('importance', ascending=False).head(20)
    
    for _, row in feat_imp.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")

    # gets it stuck in a pickle
    joblib_path = os.path.join(os.path.dirname(__file__), 'xgb_model.pkl')
    joblib.dump(model, joblib_path)
    print(f"[INFO] Pickle model saved to {joblib_path}")

if __name__ == "__main__":
    print("[START] (xgboost.train)")
    main()
    print("[DONE]")
