import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# --- Grad-CAM Class for Explainability ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x_img, x_tab, class_idx):
        self.model.zero_grad()
        logits = self.model(x_img, x_tab)
        loss = logits[0, class_idx]
        loss.backward()
        
        # Get activations and gradients
        acts = self.activations[0].detach().cpu().numpy() # (channels, h, w)
        grads = self.gradients[0].detach().cpu().numpy() # (channels, h, w)
        
        # Global average pool the gradients
        weights = np.mean(grads, axis=(1, 2)) # (channels,)
        
        # Compute weighted sum of activations
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]
            
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam
        
    def release(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

# --- Custom PyTorch Dataset ---
class CachedHybridDataset(Dataset):
    def __init__(self, indices, X_img_cached, X_tab, y):
        self.indices = indices
        self.X_img_cached = X_img_cached # Pre-computed embeddings or intermediate feature maps
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        x_img = torch.tensor(self.X_img_cached[global_idx], dtype=torch.float32)
        x_tab = self.X_tab[idx]
        label = self.y[idx]
        return x_img, x_tab, label

# --- Hybrid Model Definition ---
class HybridModel(nn.Module):
    def __init__(self, tabular_dim):
        super().__init__()
        # Image branch: EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Modify first conv layer to accept 6 channels
        old_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias
        )
        # Copy pre-trained weights to 6 channels
        with torch.no_grad():
            self.backbone.features[0][0].weight[:, :3, :, :] = old_conv.weight
            self.backbone.features[0][0].weight[:, 3:, :, :] = old_conv.weight
            
        self.backbone.classifier = nn.Identity() # outputs 1280-dim embedding
        
        # Tabular (OSM) branch
        self.tab_branch = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )
        
        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(1280 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3) # outputs raw logits for cross-entropy
        )
        
    def forward_stage1(self, img_emb, x_tab):
        """Used in Epochs 1-5 where backbone is fully frozen."""
        tab_emb = self.tab_branch(x_tab)
        x = torch.cat([img_emb, tab_emb], dim=1)
        logits = self.fusion(x)
        return logits
        
    def forward_stage2(self, intermediate_feat, x_tab):
        """Used in Epochs 6-10 with fine-tuned features[7] and features[8]."""
        # Run features block 7
        x_img = self.backbone.features[7](intermediate_feat)
        # Run features block 8
        x_img = self.backbone.features[8](x_img)
        # Run pool & flatten to get 1280-dim embedding
        x_img = self.backbone.avgpool(x_img)
        img_emb = torch.flatten(x_img, 1)
        
        tab_emb = self.tab_branch(x_tab)
        x = torch.cat([img_emb, tab_emb], dim=1)
        logits = self.fusion(x)
        return logits

    def forward(self, x_img, x_tab):
        """Standard forward pass for inference / Grad-CAM."""
        img_emb = self.backbone(x_img)
        tab_emb = self.tab_branch(x_tab)
        x = torch.cat([img_emb, tab_emb], dim=1)
        logits = self.fusion(x)
        return logits

def train_model(model_name, tabular_cols, df_train, df_val, train_indices, val_indices, all_images, y_train_enc, y_val_enc, device):
    logger.info(f"Preparing datasets for {model_name}...")
    scaler = StandardScaler()
    X_train_tab = scaler.fit_transform(df_train[tabular_cols])
    X_val_tab = scaler.transform(df_val[tabular_cols])
    
    # Initialize baseline model instance to pre-extract cached features
    model = HybridModel(tabular_dim=len(tabular_cols)).to(device)
    model.eval()
    
    # Stage 1: Extract complete image embeddings (shape: 4594, 1280)
    logger.info(f"[{model_name}] Pre-extracting image embeddings for Stage 1...")
    img_embs = []
    with torch.no_grad():
        # Process in batches of 128
        for offset in range(0, len(all_images), 128):
            batch_imgs = torch.tensor(all_images[offset:offset+128], dtype=torch.float32).to(device)
            emb = model.backbone(batch_imgs)
            img_embs.append(emb.cpu().numpy())
    img_embs = np.concatenate(img_embs, axis=0)
    
    # Stage 2: Extract intermediate feature maps after block 6 (shape: 4594, 112, 16, 16)
    logger.info(f"[{model_name}] Pre-extracting intermediate feature maps for Stage 2...")
    intermediate_feats = []
    with torch.no_grad():
        for offset in range(0, len(all_images), 128):
            batch_imgs = torch.tensor(all_images[offset:offset+128], dtype=torch.float32).to(device)
            # Forward pass through layers 0 to 6
            x = batch_imgs
            for i in range(7):
                x = model.backbone.features[i](x)
            intermediate_feats.append(x.cpu().numpy())
    intermediate_feats = np.concatenate(intermediate_feats, axis=0)
    
    # Setup initial Dataloaders for Stage 1 (using img_embs cached)
    train_dataset = CachedHybridDataset(train_indices, img_embs, X_train_tab, y_train_enc)
    val_dataset = CachedHybridDataset(val_indices, img_embs, X_val_tab, y_val_enc)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Optimize train parameters
    criterion = nn.CrossEntropyLoss()
    # In Stage 1, backbone parameters are frozen
    for param in model.backbone.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    best_f1 = 0.0
    best_model_state = None
    best_val_report = None
    best_cm = None
    
    # Run 10 Epochs
    for epoch in range(10):
        # Unfreeze and switch dataloaders for Stage 2 starting at Epoch 6 (index 5)
        if epoch == 5:
            logger.info(f"[{model_name}] Switching to Stage 2: unfreezing backbone blocks 7 & 8...")
            for param in model.backbone.features[7].parameters():
                param.requires_grad = True
            for param in model.backbone.features[8].parameters():
                param.requires_grad = True
                
            # Recreate dataloaders using intermediate_feats cached
            train_dataset = CachedHybridDataset(train_indices, intermediate_feats, X_train_tab, y_train_enc)
            val_dataset = CachedHybridDataset(val_indices, intermediate_feats, X_val_tab, y_val_enc)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            
            optimizer = optim.Adam([
                {'params': filter(lambda p: p.requires_grad and not any(id(p) == id(bp) for bp in list(model.backbone.features[7].parameters()) + list(model.backbone.features[8].parameters())), model.parameters()), 'lr': 1e-3},
                {'params': list(model.backbone.features[7].parameters()) + list(model.backbone.features[8].parameters()), 'lr': 1e-5}
            ])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
            
        model.train()
        train_loss = 0.0
        for x_img, x_tab, y in train_loader:
            x_img, x_tab, y = x_img.to(device), x_tab.to(device), y.to(device)
            optimizer.zero_grad()
            
            if epoch < 5:
                logits = model.forward_stage1(x_img, x_tab)
            else:
                logits = model.forward_stage2(x_img, x_tab)
                
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_img.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x_img, x_tab, y in val_loader:
                x_img, x_tab, y = x_img.to(device), x_tab.to(device), y.to(device)
                if epoch < 5:
                    logits = model.forward_stage1(x_img, x_tab)
                else:
                    logits = model.forward_stage2(x_img, x_tab)
                    
                loss = criterion(logits, y)
                val_loss += loss.item() * x_img.size(0)
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        acc = accuracy_score(all_targets, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
        
        logger.info(f"[{model_name}] Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
        
        # Save best model checkpoint
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = pickle.dumps(model.state_dict())
            best_val_report = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_macro": f1,
                "val_loss": val_loss,
                "train_loss": train_loss
            }
            best_cm = confusion_matrix(all_targets, all_preds)
            
    best_model = HybridModel(tabular_dim=len(tabular_cols)).to(device)
    best_model.load_state_dict(pickle.loads(best_model_state))
    return best_model, best_val_report, best_cm, scaler

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    cnn_dir = data_dir / "cnn"
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    reports_dir = project_root / "reports"
    
    # Load dataset index
    df_cnn = pd.read_csv(cnn_dir / "cnn_dataset.csv")
    
    # Pre-load all 4,594 Sentinel-2 image patches into memory
    logger.info("Pre-loading all 4,594 Sentinel-2 image patches into memory...")
    all_images = np.zeros((len(df_cnn), 6, 128, 128), dtype=np.float32)
    for idx, row in df_cnn.iterrows():
        img_rel_path = row["image_path"]
        img_abs_path = project_root / img_rel_path
        all_images[idx] = np.load(img_abs_path)
    logger.info("Pre-loading completed successfully.")
    
    # Encode Target classes (High = 0, Low = 1, Medium = 2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(df_cnn["growth_category"])
    
    indices = np.arange(len(df_cnn))
    train_indices, val_indices, y_train_enc, y_val_enc = train_test_split(
        indices, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    df_train = df_cnn.iloc[train_indices].reset_index(drop=True)
    df_val = df_cnn.iloc[val_indices].reset_index(drop=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Feature columns subsets
    features_osm_only = [
        "building_count", "building_density", "road_density",
        "road_intersection_count", "green_ratio", "distance_to_center", "distance_to_highway"
    ]
    features_osm_spectral = features_osm_only + ["mean_ndvi", "mean_ndbi", "mean_ndwi"]
    
    # Train Hybrid A (OSM only)
    logger.info("Starting training of Hybrid A (OSM Tabular branch)...")
    model_a, report_a, cm_a, scaler_a = train_model(
        "Hybrid A", features_osm_only, df_train, df_val, train_indices, val_indices, all_images, y_train_enc, y_val_enc, device
    )
    
    # Train Hybrid B (OSM + Spectral)
    logger.info("Starting training of Hybrid B (OSM + Spectral Tabular branch)...")
    model_b, report_b, cm_b, scaler_b = train_model(
        "Hybrid B", features_osm_spectral, df_train, df_val, train_indices, val_indices, all_images, y_train_enc, y_val_enc, device
    )
    
    # Save the best model of the two
    if report_b["f1_macro"] > report_a["f1_macro"]:
        best_model = model_b
        best_report = report_b
        best_cm = cm_b
        best_scaler = scaler_b
        best_cols = features_osm_spectral
        best_name = "Hybrid B (OSM + Spectral)"
    else:
        best_model = model_a
        best_report = report_a
        best_cm = cm_a
        best_scaler = scaler_a
        best_cols = features_osm_only
        best_name = "Hybrid A (OSM only)"
        
    logger.info(f"Saving best model: {best_name} (Val F1 Macro: {best_report['f1_macro']:.4f})")
    torch.save(best_model.state_dict(), models_dir / "hybrid_cnn_osm_model.pth")
    
    # Save best metrics JSON
    with open(results_dir / "hybrid_metrics.json", "w") as f:
        json.dump({
            "model_name": best_name,
            "accuracy": float(best_report["accuracy"]),
            "precision_macro": float(best_report["precision"]),
            "recall_macro": float(best_report["recall"]),
            "f1_macro": float(best_report["f1_macro"]),
            "confusion_matrix": best_cm.tolist()
        }, f, indent=4)
        
    # Plot best model Confusion Matrix
    classes = ["High", "Low", "Medium"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Oranges", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix: {best_name}")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix_hybrid.png", dpi=150)
    plt.close()
    
    # --- Generate Grad-CAM Explainability ---
    logger.info("Generating Grad-CAM overlays...")
    best_model.eval()
    gradcam = GradCAM(best_model, best_model.backbone.features[-1])
    
    # Scale tabular validation features
    X_val_tab_scaled = best_scaler.transform(df_val[best_cols])
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    categories = ["High", "Low", "Medium"]
    
    for idx, (cat_name, cat_code) in enumerate(zip(categories, [0, 1, 2])):
        # Find matches for target category in validation set
        matches = np.where(y_val_enc == cat_code)[0]
        sample_idx = matches[0]
        
        # Load image patch manually for Grad-CAM full forward pass
        img_rel_path = df_val.iloc[sample_idx]["image_path"]
        img_abs_path = project_root / img_rel_path
        patch = np.load(img_abs_path)
        x_img = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
        x_tab = torch.tensor(X_val_tab_scaled[sample_idx], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Compute Grad-CAM
        cam = gradcam(x_img, x_tab, cat_code)
        
        # Display original RGB
        rgb = patch[[2, 1, 0], :, :]
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb_min = rgb.min(axis=(0, 1), keepdims=True)
        rgb_max = rgb.max(axis=(0, 1), keepdims=True)
        rgb_scaled = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
        rgb_scaled = np.clip(rgb_scaled, 0.0, 1.0)
        
        axes[idx, 0].imshow(rgb_scaled)
        axes[idx, 0].set_title(f"Original RGB - {cat_name} Growth")
        axes[idx, 0].axis("off")
        
        # Overlay Grad-CAM
        axes[idx, 1].imshow(rgb_scaled)
        axes[idx, 1].imshow(cam, cmap="jet", alpha=0.5)
        axes[idx, 1].set_title(f"Grad-CAM Heatmap - {cat_name} Growth")
        axes[idx, 1].axis("off")
        
    gradcam.release()
    plt.tight_layout()
    plt.savefig(reports_dir / "gradcam_examples.png", dpi=150)
    plt.close()
    
    # --- Load Baseline models and Compile final model_comparison_final.csv ---
    logger.info("Compiling model performance comparison CSV...")
    
    rf_metrics = {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}
    rf_path = results_dir / "random_forest_metrics.json"
    if rf_path.exists():
        with open(rf_path, "r") as f:
            rf_metrics = json.load(f)
            
    xgb_metrics = {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}
    xgb_path = results_dir / "xgboost_metrics.json"
    if xgb_path.exists():
        with open(xgb_path, "r") as f:
            xgb_metrics = json.load(f)
            
    xgb_tuned_metrics = {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}
    xgb_tuned_path = results_dir / "xgboost_tuned_metrics.json"
    if xgb_tuned_path.exists():
        with open(xgb_tuned_path, "r") as f:
            xgb_tuned_metrics = json.load(f)
            
    comparison_records = [
        {
            "Model": "Random Forest",
            "Accuracy": rf_metrics["accuracy"],
            "Precision (macro)": rf_metrics["precision_macro"],
            "Recall (macro)": rf_metrics["recall_macro"],
            "F1 Score (macro)": rf_metrics["f1_macro"]
        },
        {
            "Model": "Baseline XGBoost",
            "Accuracy": xgb_metrics["accuracy"],
            "Precision (macro)": xgb_metrics["precision_macro"],
            "Recall (macro)": xgb_metrics["recall_macro"],
            "F1 Score (macro)": xgb_metrics["f1_macro"]
        },
        {
            "Model": "Tuned XGBoost",
            "Accuracy": xgb_tuned_metrics["accuracy"],
            "Precision (macro)": xgb_tuned_metrics["precision_macro"],
            "Recall (macro)": xgb_tuned_metrics["recall_macro"],
            "F1 Score (macro)": xgb_tuned_metrics["f1_macro"]
        },
        {
            "Model": "Hybrid A (OSM only)",
            "Accuracy": report_a["accuracy"],
            "Precision (macro)": report_a["precision"],
            "Recall (macro)": report_a["recall"],
            "F1 Score (macro)": report_a["f1_macro"]
        },
        {
            "Model": "Hybrid B (OSM + Spectral)",
            "Accuracy": report_b["accuracy"],
            "Precision (macro)": report_b["precision"],
            "Recall (macro)": report_b["recall"],
            "F1 Score (macro)": report_b["f1_macro"]
        }
    ]
    
    df_compare = pd.DataFrame(comparison_records)
    df_compare.to_csv(results_dir / "model_comparison_final.csv", index=False)
    
    # 7. Print Console Outputs
    print("\n" + "="*60)
    print("           HYBRID DEEP LEARNING MODEL RESULTS")
    print("="*60)
    print(f"Hybrid A (OSM only) Val Accuracy       : {report_a['accuracy']:.4f}")
    print(f"Hybrid A (OSM only) Val F1 Macro       : {report_a['f1_macro']:.4f}")
    print()
    print(f"Hybrid B (OSM + Spectral) Val Accuracy : {report_b['accuracy']:.4f}")
    print(f"Hybrid B (OSM + Spectral) Val F1 Macro : {report_b['f1_macro']:.4f}")
    print()
    print("Best Performing Hybrid Model:")
    print(f"  - Model Name: {best_name}")
    print(f"  - Validation Loss: {best_report['val_loss']:.4f}")
    print(f"  - Validation Accuracy: {best_report['accuracy']:.4f}")
    print(f"  - Macro Precision: {best_report['precision']:.4f}")
    print(f"  - Macro Recall   : {best_report['recall']:.4f}")
    print(f"  - Macro F1       : {best_report['f1_macro']:.4f}")
    print("="*60 + "\n")
    
    print("="*60)
    print("         FINAL MODEL COMPARISON (ALL ARCHITECTURES)")
    print("="*60)
    print(df_compare.to_string(index=False))
    print("="*60 + "\n")
    
    # Recommendation Section in Console
    print("Recommendation Summary:")
    diff = best_report['f1_macro'] - xgb_metrics['f1_macro']
    if diff > 0.005:
        print(f"  - VERDICT: YES, the Hybrid model improves performance by {diff:+.4f} F1 macro.")
    else:
        print(f"  - VERDICT: NO, classical XGBoost remains the optimal choice (difference: {diff:+.4f} F1 macro).")
    print()

if __name__ == "__main__":
    main()
