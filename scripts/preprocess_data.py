import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from loguru import logger

def main():
    # 1. Setup paths
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "features" / "combined_growth_dataset.csv"
    ml_dir = project_root / "data" / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading growth score dataset...")
    df = pd.read_csv(data_path)
    
    # 2. Drop identifier column
    if "grid_id" in df.columns:
        logger.info("Dropping grid_id identifier column...")
        df = df.drop(columns=["grid_id"])
        
    # 3. Apply One-Hot Encoding for city column
    logger.info("One-hot encoding categorical 'city' column...")
    df = pd.get_dummies(df, columns=["city"], dtype=int)
    
    # 4. Correlation analysis on input numerical features
    # Exclude change_category (categorical target) and urban_change_index (regression target, not input)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "urban_change_index" in num_cols:
        num_cols.remove("urban_change_index")
    # Also exclude one-hot encoded city columns from feature correlation drop check
    num_cols = [c for c in num_cols if not c.startswith("city_")]
    
    corr_matrix = df[num_cols].corr().abs()
    
    # Identify pairs with absolute correlation > 0.95 and select one to drop
    dropped_features = {}
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            col1 = corr_matrix.columns[j]
            col2 = corr_matrix.columns[i]
            val = corr_matrix.loc[col1, col2]
            if val > 0.95:
                # If neither is already dropped, drop the second column (col2)
                if col2 not in dropped_features and col1 not in dropped_features.values():
                    dropped_features[col2] = (col1, val)
                    
    logger.info(f"Identified {len(dropped_features)} highly correlated features (|r| > 0.95):")
    for feat, (ref, val) in dropped_features.items():
        print(f"  - Removed '{feat}' -> Highly correlated with '{ref}' (r = {val:.4f})")
        
    # Drop the highly correlated features
    df_filtered = df.drop(columns=list(dropped_features.keys()))
    
    # 5. Separate features X and target y
    X = df_filtered.drop(columns=["change_category"])
    y = df_filtered[["change_category"]].copy()
    
    # Encode target change_category using LabelEncoder
    logger.info("Label encoding change_category...")
    le = LabelEncoder()
    y["change_category"] = le.fit_transform(y["change_category"])
    
    # Print the label encoding mapping
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    logger.info(f"Target Label Mapping: {mapping}")
    
    # 6. Stratified Train-Test Split (80/20, Stratify by target y)
    logger.info("Splitting dataset into stratified train and test partitions (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 7. Standardize input numerical features
    # Select input numerical features (excluding urban_change_index and one-hot encoded city columns)
    input_numeric_features = [
        col for col in X_train.columns
        if col != "urban_change_index" and not col.startswith("city_")
    ]
    
    logger.info("Standardizing input numerical feature columns...")
    scaler = StandardScaler()
    
    # Fit on training and transform both
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[input_numeric_features] = scaler.fit_transform(X_train[input_numeric_features])
    X_test_scaled[input_numeric_features] = scaler.transform(X_test[input_numeric_features])
    
    # Save the fitted scaler
    scaler_path = ml_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved fitted StandardScaler to: {scaler_path}")
    
    # 8. Save train/test datasets
    logger.info("Saving split CSV datasets...")
    X_train_scaled.to_csv(ml_dir / "X_train.csv", index=False)
    X_test_scaled.to_csv(ml_dir / "X_test.csv", index=False)
    y_train.to_csv(ml_dir / "y_train.csv", index=False)
    y_test.to_csv(ml_dir / "y_test.csv", index=False)
    
    # 9. Print Preprocessing Summary
    print("\n" + "="*50)
    print("           PREPROCESSING PIPELINE SUMMARY")
    print("="*50)
    print(f"Number of Training Samples: {len(X_train_scaled)}")
    print(f"Number of Testing Samples : {len(X_test_scaled)}")
    print()
    print("Final Feature List:")
    for idx, col in enumerate(X_train_scaled.columns):
        # Indicate unscaled columns for clarity
        note = " (Kept unscaled)" if col in ["urban_change_index"] or col.startswith("city_") else ""
        print(f"  {idx+1:2d}. {col}{note}")
    print()
    print("Removed Correlated Features:")
    for feat, (ref, val) in dropped_features.items():
        print(f"  - {feat} (due to high correlation with {ref}: r = {val:.4f})")
    print()
    
    # Class distribution
    train_dist = y_train["change_category"].value_counts().sort_index().to_dict()
    test_dist = y_test["change_category"].value_counts().sort_index().to_dict()
    
    # Reverse classes for display
    class_names = list(le.classes_)
    print("Class Distribution in Train:")
    for code, count in train_dist.items():
        print(f"  - {class_names[code]} (code {code}): {count} ({count/len(y_train)*100:.2f}%)")
    print()
    print("Class Distribution in Test:")
    for code, count in test_dist.items():
        print(f"  - {class_names[code]} (code {code}): {count} ({count/len(y_test)*100:.2f}%)")
    print("="*50 + "\n")
    
    logger.info("Preprocessing complete.")

if __name__ == "__main__":
    main()
