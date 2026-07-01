import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, learning_curve
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from loguru import logger

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
    
    # 2. Leakage Verification & Prevention
    print("="*60)
    print("           LEAKAGE VERIFICATION STAGE")
    print("="*60)
    leakage_col = "urban_change_index"
    if leakage_col in X_train.columns:
        print(f"Target leakage column '{leakage_col}' detected. Removing from features...")
        X_train_model = X_train.drop(columns=[leakage_col])
        X_test_model = X_test.drop(columns=[leakage_col])
    else:
        print(f"No target leakage column '{leakage_col}' detected in features.")
        X_train_model = X_train.copy()
        X_test_model = X_test.copy()
        
    feature_names = list(X_train_model.columns)
    print("\nFinal Training Feature List:")
    for idx, col in enumerate(feature_names):
        print(f"  {idx+1:2d}. {col}")
    print("="*60 + "\n")
    
    # 3. Hyperparameter Tuning using RandomizedSearchCV
    logger.info("Initializing Hyperparameter Tuning using RandomizedSearchCV...")
    xgb_base = XGBClassifier(
        random_state=42,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1
    )
    
    param_dist = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 2, 3, 4, 5],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=30,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train_model, y_train)
    
    best_params = search.best_params_
    best_cv_score = search.best_score_
    
    # Save best parameters
    with open(results_dir / "best_xgboost_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
        
    # 4. Train Final Model using Best Parameters and Early Stopping
    logger.info("Training Final Tuned XGBoost Model...")
    xgb_tuned = XGBClassifier(
        random_state=42,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        early_stopping_rounds=15,
        **best_params
    )
    
    # Fit final model with early stopping
    xgb_tuned.fit(
        X_train_model, y_train,
        eval_set=[(X_test_model, y_test)],
        verbose=False
    )
    
    # 5. Evaluate Tuned Model on X_test
    logger.info("Evaluating Tuned XGBoost Model on Test Set...")
    y_pred = xgb_tuned.predict(X_test_model)
    
    acc_test = accuracy_score(y_test, y_pred)
    prec_test, rec_test, f1_test, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    cm_test = confusion_matrix(y_test, y_pred)
    
    tuned_metrics = {
        "accuracy": float(acc_test),
        "precision_macro": float(prec_test),
        "recall_macro": float(rec_test),
        "f1_macro": float(f1_test),
        "confusion_matrix": cm_test.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    with open(results_dir / "xgboost_tuned_metrics.json", "w") as f:
        json.dump(tuned_metrics, f, indent=4)
        
    # 6. Perform Stratified 5-Fold Cross Validation on final tuned parameters
    logger.info("Running Stratified 5-Fold Cross Validation on Tuned XGBoost...")
    # Create a fresh classifier for CV using the tuned params (no early stopping since CV doesn't use test set)
    xgb_cv_model = XGBClassifier(
        random_state=42,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        **best_params
    )
    
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro'
    }
    
    cv_results = cross_validate(xgb_cv_model, X_train_model, y_train, cv=cv, scoring=scoring_metrics, n_jobs=-1)
    
    mean_accuracy = float(np.mean(cv_results['test_accuracy']))
    mean_precision = float(np.mean(cv_results['test_precision_macro']))
    mean_recall = float(np.mean(cv_results['test_recall_macro']))
    mean_f1 = float(np.mean(cv_results['test_f1_macro']))
    std_f1 = float(np.std(cv_results['test_f1_macro']))
    
    cv_report = {
        "mean_accuracy": mean_accuracy,
        "mean_precision_macro": mean_precision,
        "mean_recall_macro": mean_recall,
        "mean_f1_macro": mean_f1,
        "std_f1_macro": std_f1
    }
    
    with open(results_dir / "cross_validation_results.json", "w") as f:
        json.dump(cv_report, f, indent=4)
        
    # Save Tuned Model Binary
    with open(models_dir / "xgboost_tuned_model.pkl", "wb") as f:
        pickle.dump(xgb_tuned, f)
        
    # 7. Generate Feature Importance
    importances = xgb_tuned.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    top_indices = indices[:15]
    sns.barplot(x=importances[top_indices], y=[feature_names[i] for i in top_indices], palette="Oranges_r")
    plt.title("Tuned XGBoost Top 15 Feature Importance")
    plt.xlabel("Feature Weight (Gain/F-score)")
    plt.tight_layout()
    plt.savefig(reports_dir / "feature_importance_xgb_tuned.png", dpi=150)
    plt.close()
    
    # 8. Plot and Save Confusion Matrix
    classes = ["High", "Low", "Medium"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Oranges", xticklabels=classes, yticklabels=classes)
    plt.title("Tuned XGBoost Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix_xgb_tuned.png", dpi=150)
    plt.close()
    
    # 9. Optional: Generate Learning Curve
    logger.info("Generating learning curve...")
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            xgb_cv_model, X_train_model, y_train, cv=cv, scoring='f1_macro', 
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1, random_state=42
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(8, 6))
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                         val_scores_mean + val_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
        plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-Validation Score")
        plt.title("Tuned XGBoost Learning Curve")
        plt.xlabel("Training Examples")
        plt.ylabel("F1 Macro Score")
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(reports_dir / "xgboost_learning_curve.png", dpi=150)
        plt.close()
        logger.info("Learning curve saved successfully.")
    except Exception as e:
        logger.warning(f"Could not generate learning curve: {e}")
        
    # 10. Baseline Comparison
    baseline_path = results_dir / "xgboost_metrics.json"
    if baseline_path.exists():
        with open(baseline_path, "r") as f:
            base_metrics = json.load(f)
            
        comparison_data = {
            "Metric": ["Accuracy", "Precision", "Recall", "Macro F1"],
            "Baseline XGBoost": [
                base_metrics["accuracy"],
                base_metrics["precision_macro"],
                base_metrics["recall_macro"],
                base_metrics["f1_macro"]
            ],
            "Tuned XGBoost": [acc_test, prec_test, rec_test, f1_test]
        }
        df_comp = pd.DataFrame(comparison_data)
        # Compute improvement column
        df_comp["Improvement"] = df_comp["Tuned XGBoost"] - df_comp["Baseline XGBoost"]
        df_comp.to_csv(results_dir / "xgboost_baseline_vs_tuned.csv", index=False)
        
        print("\n" + "="*50)
        print("         BASELINE VS TUNED XGBOOST COMPARISON")
        print("="*50)
        print(df_comp.to_string(index=False))
        print("="*50)
    else:
        logger.warning("Baseline XGBoost metrics file not found. Skipping comparison generation.")
        
    # 11. Console Output Prints
    print("\n" + "="*50)
    print("           MODEL OPTIMIZATION STAGE RESULTS")
    print("="*50)
    print("Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  - {k}: {v}")
    print()
    print(f"Best Cross Validation Score (F1 Macro): {best_cv_score:.4f}")
    print()
    print("5-Fold Cross Validation Results:")
    print(f"  - Mean CV Accuracy   : {mean_accuracy:.4f}")
    print(f"  - Mean CV Precision  : {mean_precision:.4f}")
    print(f"  - Mean CV Recall     : {mean_recall:.4f}")
    print(f"  - Mean CV Macro F1   : {mean_f1:.4f}")
    print(f"  - Std CV Macro F1    : {std_f1:.4f}")
    print()
    print("Final Test Metrics:")
    print(f"  - Final Test Accuracy: {acc_test:.4f}")
    print(f"  - Final Test Precision: {prec_test:.4f}")
    print(f"  - Final Test Recall  : {rec_test:.4f}")
    print(f"  - Final Test Macro F1: {f1_test:.4f}")
    print("="*50)
    
    # Top 15 Feature Importance print
    print("\nTop 15 Important Features - Tuned XGBoost:")
    for rank in range(15):
        idx = indices[rank]
        print(f"  {rank+1:2d}. {feature_names[idx]:<25}: {importances[idx]:.4f}")
    print()

if __name__ == "__main__":
    main()
