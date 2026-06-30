import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def main():
    # 1. Setup paths
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "ml"
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    reports_dir = project_root / "reports"
    
    # Create directories
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset partitions
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    # 2. Leakage Prevention: Drop urban_change_index from training/testing features
    # Keeping city columns and other features
    if "urban_change_index" in X_train.columns:
        X_train_model = X_train.drop(columns=["urban_change_index"])
        X_test_model = X_test.drop(columns=["urban_change_index"])
    else:
        X_train_model = X_train.copy()
        X_test_model = X_test.copy()
        
    feature_names = list(X_train_model.columns)
    
    # 3. Model 1: Random Forest Classifier
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_model.fit(X_train_model, y_train)
    
    # Predict on test set
    y_pred_rf = rf_model.predict(X_test_model)
    
    # 4. Model 2: XGBoost Classifier
    print("Training XGBoost...")
    xgb_model = XGBClassifier(
        random_state=42,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1
    )
    xgb_model.fit(X_train_model, y_train)
    
    # Predict on test set
    y_pred_xgb = xgb_model.predict(X_test_model)
    
    print("Evaluation Complete.")
    
    # 5. Evaluate Random Forest
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, average="macro")
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    
    rf_metrics = {
        "accuracy": float(acc_rf),
        "precision_macro": float(prec_rf),
        "recall_macro": float(rec_rf),
        "f1_macro": float(f1_rf),
        "confusion_matrix": cm_rf.tolist(),
        "classification_report": classification_report(y_test, y_pred_rf, output_dict=True)
    }
    
    with open(results_dir / "random_forest_metrics.json", "w") as f:
        json.dump(rf_metrics, f, indent=4)
        
    # 6. Evaluate XGBoost
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    prec_xgb, rec_xgb, f1_xgb, _ = precision_recall_fscore_support(y_test, y_pred_xgb, average="macro")
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    
    xgb_metrics = {
        "accuracy": float(acc_xgb),
        "precision_macro": float(prec_xgb),
        "recall_macro": float(rec_xgb),
        "f1_macro": float(f1_xgb),
        "confusion_matrix": cm_xgb.tolist(),
        "classification_report": classification_report(y_test, y_pred_xgb, output_dict=True)
    }
    
    with open(results_dir / "xgboost_metrics.json", "w") as f:
        json.dump(xgb_metrics, f, indent=4)
        
    # 7. Compare Models
    comparison_data = {
        "Model": ["Random Forest", "XGBoost"],
        "Accuracy": [acc_rf, acc_xgb],
        "Precision": [prec_rf, prec_xgb],
        "Recall": [rec_rf, rec_xgb],
        "F1 Score": [f1_rf, f1_xgb]
    }
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    
    # Display comparison table
    print("\n" + "="*50)
    print("             MODEL COMPARISON TABLE")
    print("="*50)
    print(df_comparison.to_string(index=False))
    print("="*50)
    
    # Identify Best Model based on F1 Score
    best_row = df_comparison.loc[df_comparison["F1 Score"].idxmax()]
    best_model_name = best_row["Model"]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Accuracy: {best_row['Accuracy']:.4f}")
    print(f"Macro Precision: {best_row['Precision']:.4f}")
    print(f"Macro Recall: {best_row['Recall']:.4f}")
    print(f"Macro F1 Score: {best_row['F1 Score']:.4f}")
    print()
    
    # 8. Save Models
    with open(models_dir / "random_forest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open(models_dir / "xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
        
    # 9. Plot and Save Confusion Matrices
    classes = ["High", "Low", "Medium"]
    
    # RF Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Random Forest Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix_rf.png", dpi=150)
    plt.close()
    
    # XGB Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Oranges", xticklabels=classes, yticklabels=classes)
    plt.title("XGBoost Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix_xgb.png", dpi=150)
    plt.close()
    
    # 10. Feature Importance Analysis
    importances_rf = rf_model.feature_importances_
    indices_rf = np.argsort(importances_rf)[::-1]
    
    importances_xgb = xgb_model.feature_importances_
    indices_xgb = np.argsort(importances_xgb)[::-1]
    
    # Display Top 15 Features for both models
    print("\nTop 15 Important Features - Random Forest:")
    for rank in range(15):
        idx = indices_rf[rank]
        print(f"  {rank+1:2d}. {feature_names[idx]:<25}: {importances_rf[idx]:.4f}")
        
    print("\nTop 15 Important Features - XGBoost:")
    for rank in range(15):
        idx = indices_xgb[rank]
        print(f"  {rank+1:2d}. {feature_names[idx]:<25}: {importances_xgb[idx]:.4f}")
        
    # Plot RF Feature Importance
    plt.figure(figsize=(10, 6))
    top_indices_rf = indices_rf[:15]
    sns.barplot(x=importances_rf[top_indices_rf], y=[feature_names[i] for i in top_indices_rf], palette="Blues_r")
    plt.title("Random Forest Top 15 Feature Importance")
    plt.xlabel("Mean Decrease in Impurity")
    plt.tight_layout()
    plt.savefig(reports_dir / "feature_importance_rf.png", dpi=150)
    plt.close()
    
    # Plot XGB Feature Importance
    plt.figure(figsize=(10, 6))
    top_indices_xgb = indices_xgb[:15]
    sns.barplot(x=importances_xgb[top_indices_xgb], y=[feature_names[i] for i in top_indices_xgb], palette="Oranges_r")
    plt.title("XGBoost Top 15 Feature Importance")
    plt.xlabel("Feature Weight (Gain/F-score)")
    plt.tight_layout()
    plt.savefig(reports_dir / "feature_importance_xgb.png", dpi=150)
    plt.close()
    
    print("\nModel training, evaluation, plots, and metrics files generated successfully.")

if __name__ == "__main__":
    main()
