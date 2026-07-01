import pickle
import pandas as pd
import shap
from pathlib import Path

def main():
    project_root = Path(".").resolve()
    model_path = project_root / "models" / "xgboost_model.pkl"
    data_path = project_root / "data" / "ml" / "X_test.csv"
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    X_test = pd.read_csv(data_path).drop(columns=["urban_change_index"])
    
    print("X_test shape:", X_test.shape)
    
    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    print("TreeExplainer base values shape:", explainer.expected_value)
    if isinstance(shap_values, list):
        print(f"SHAP values is a list of length {len(shap_values)}. Element shape: {shap_values[0].shape}")
    else:
        print("SHAP values shape:", shap_values.shape)

if __name__ == "__main__":
    main()
