import os
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
import json

from ml.chest_ai.config import settings, Settings
from ml.chest_ai.dataloader import get_dataloaders
from ml.chest_ai.model import ChestClassifier
from ml.chest_ai.checkpoint import CheckpointManager
from ml.chest_ai.metrics import compute_metrics
from ml.chest_ai.gradcam import GradCAM
from ml.chest_ai.visualization import overlay_heatmap, plot_prediction_with_gradcam, denormalize_image
from ml.chest_ai.logger import logger

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ChestDiseaseAI model and generate explainability outputs.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file.")
    parser.add_argument("--output-dir", type=str, default="./outputs/evaluation", help="Directory to save evaluation assets.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for metrics.")
    return parser.parse_args()

def generate_curves_and_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: str
):
    """
    Plots and saves ROC Curves, Precision-Recall Curves, and Confusion Matrices.
    Handles -1.0 ignore targets robustly.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_classes = len(class_names)
    
    # 1. ROC Curves
    plt.figure(figsize=(8, 6))
    for c_idx, name in enumerate(class_names):
        true_c = y_true[:, c_idx]
        pred_c = y_pred[:, c_idx]
        
        valid = true_c != -1.0
        if len(np.unique(true_c[valid])) == 2:
            fpr, tpr, _ = roc_curve(true_c[valid], pred_c[valid])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
            
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Label Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(roc_path, bbox_inches="tight", dpi=150)
    plt.close()
    
    # 2. PR Curves
    plt.figure(figsize=(8, 6))
    for c_idx, name in enumerate(class_names):
        true_c = y_true[:, c_idx]
        pred_c = y_pred[:, c_idx]
        
        valid = true_c != -1.0
        if len(np.unique(true_c[valid])) == 2:
            precision, recall, _ = precision_recall_curve(true_c[valid], pred_c[valid])
            ap = average_precision_score(true_c[valid], pred_c[valid])
            plt.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")
            
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multi-Label Precision-Recall (PR) Curves")
    plt.legend(loc="lower left")
    pr_path = os.path.join(output_dir, "pr_curves.png")
    plt.savefig(pr_path, bbox_inches="tight", dpi=150)
    plt.close()
    
    # 3. Confusion Matrices
    cols = 3 if num_classes >= 3 else num_classes
    rows = int(np.ceil(num_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    if num_classes == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
        
    for c_idx, name in enumerate(class_names):
        true_c = y_true[:, c_idx]
        pred_c = y_pred[:, c_idx]
        
        valid = true_c != -1.0
        true_clean = true_c[valid]
        pred_binary = (pred_c[valid] >= 0.5).astype(np.float32)
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_clean, pred_binary, labels=[0, 1])
        
        ax = axes[c_idx]
        # Custom render confusion matrix since seaborn is not in requirements
        ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"Confusion Matrix: {name}", fontsize=10, fontweight="bold")
        
        # Labels and ticks
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(["Neg", "Pos"])
        
        # Write text values inside matrix
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )
                
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        
    # Hide empty subplots
    for j in range(num_classes, len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close()

def generate_html_dashboard(
    metrics: Dict[str, Any],
    class_names: List[str],
    errors_list: List[Dict[str, Any]],
    output_dir: str
):
    """
    Generates a stunning, self-contained HTML Dashboard for error triage and explainability.
    """
    # Create HTML content
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ChestDiseaseAI - Error Analysis & Metrics Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #38bdf8;
            border-bottom: 2px solid #334155;
            padding-bottom: 10px;
        }}
        .grid-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #94a3b8;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #38bdf8;
        }}
        .table-container {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #334155;
        }}
        th {{
            color: #38bdf8;
            font-weight: 600;
        }}
        tr:hover {{
            background: #334155;
        }}
        .grid-charts {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        @media (max-width: 768px) {{
            .grid-charts {{
                grid-template-columns: 1fr;
            }}
        }}
        .chart-card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .chart-card img {{
            max-width: 100%;
            border-radius: 4px;
        }}
        .error-section {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .error-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .error-card {{
            background: #0f172a;
            border: 1px solid #475569;
            border-radius: 8px;
            padding: 15px;
        }}
        .error-card img {{
            width: 100%;
            border-radius: 4px;
            margin-bottom: 10px;
        }}
        .error-meta {{
            font-size: 13px;
            color: #94a3b8;
            line-height: 1.5;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 11px;
        }}
        .badge-fp {{
            background-color: #ef4444;
            color: #ffffff;
        }}
        .badge-fn {{
            background-color: #f59e0b;
            color: #ffffff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ChestDiseaseAI - Performance & Explainability Dashboard</h1>
        
        <!-- Summary Cards -->
        <div class="grid-stats">
            <div class="stat-card">
                <h3>Macro AUROC</h3>
                <div class="value">{metrics["macro_auroc"]:.4f}</div>
            </div>
            <div class="stat-card">
                <h3>Macro F1-Score</h3>
                <div class="value">{metrics["macro_f1"]:.4f}</div>
            </div>
            <div class="stat-card">
                <h3>Macro Accuracy</h3>
                <div class="value">{metrics["macro_accuracy"]:.4f}</div>
            </div>
            <div class="stat-card">
                <h3>Macro Precision</h3>
                <div class="value">{metrics["macro_precision"]:.4f}</div>
            </div>
        </div>

        <!-- Class Breakdown Table -->
        <div class="table-container">
            <h2>Disease Classification Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>Disease Class</th>
                        <th>AUROC</th>
                        <th>F1-Score</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>Accuracy</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add rows to table
    for name in class_names:
        c_m = metrics["class_metrics"][name]
        auroc_str = f"{c_m['auroc']:.4f}" if c_m["auroc"] is not None else "N/A"
        html_template += f"""
                    <tr>
                        <td><strong>{name}</strong></td>
                        <td>{auroc_str}</td>
                        <td>{c_m["f1"]:.4f}</td>
                        <td>{c_m["precision"]:.4f}</td>
                        <td>{c_m["recall"]:.4f}</td>
                        <td>{c_m["accuracy"]:.4f}</td>
                    </tr>
        """
        
    html_template += """
                </tbody>
            </table>
        </div>

        <!-- Performance Curves -->
        <div class="grid-charts">
            <div class="chart-card">
                <h3>Receiver Operating Characteristic (ROC) Curves</h3>
                <img src="roc_curves.png" alt="ROC Curves">
            </div>
            <div class="chart-card">
                <h3>Precision-Recall (PR) Curves</h3>
                <img src="pr_curves.png" alt="PR Curves">
            </div>
        </div>
        
        <div class="chart-card" style="margin-bottom: 30px;">
            <h3>Class Confusion Matrices</h3>
            <img src="confusion_matrices.png" alt="Confusion Matrices">
        </div>

        <!-- Error Analysis and GradCAM Section -->
        <div class="error-section">
            <h2>Error Analysis & Grad-CAM Heatmap Overlays</h2>
            <p>Visualizing misclassified samples alongside model focus areas. Top false positives and false negatives triaged below.</p>
            <div class="error-grid">
    """
    
    # Add error cards
    for err in errors_list:
        badge_class = "badge-fp" if err["error_type"] == "False Positive" else "badge-fn"
        # Compile predictions list
        pred_lines = []
        for c_idx, name in enumerate(class_names):
            pred_lines.append(f"{name}: {err['pred_probs'][c_idx]:.1%}")
        pred_text = ", ".join(pred_lines)
        
        img_rel_path = os.path.relpath(err["overlay_path"], output_dir)
        
        html_template += f"""
                <div class="error-card">
                    <img src="{img_rel_path}" alt="Error Overlay">
                    <div class="error-meta">
                        <span class="badge {badge_class}">{err["error_type"]}</span><br>
                        <strong>Target Disease:</strong> {err["target_class"]}<br>
                        <strong>File Path:</strong> <code style="word-break: break-all;">{os.path.basename(err["image_path"])}</code><br>
                        <strong>True Labels:</strong> {err["true_labels_str"]}<br>
                        <strong>Predictions:</strong> <span style="font-size: 11px;">{pred_text}</span>
                    </div>
                </div>
        """
        
    html_template += """
            </div>
        </div>
    </div>
</body>
</html>
    """
    
    html_path = os.path.join(output_dir, "dashboard.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    logger.info(f"Generated HTML Error Analysis Dashboard at: {html_path}")

def run_evaluation():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Settings
    logger.info("Setting up settings configuration...")
    # Point data_dir dynamically from settings or override if needed
    test_settings = settings
    test_settings.training.checkpoint_dir = os.path.dirname(args.checkpoint)
    
    # 2. Load Dataloaders (Val split is evaluated)
    logger.info("Setting up validation dataloaders...")
    _, val_loader, pos_weights = get_dataloaders(test_settings, num_workers=0)
    
    # 3. Load Model
    num_classes = len(test_settings.data.target_labels)
    model = ChestClassifier(
        backbone=test_settings.model.backbone,
        num_classes=num_classes,
        pretrained=False
    )
    
    # 4. Load Checkpoint weights
    manager = CheckpointManager(test_settings.training.checkpoint_dir)
    state = manager.load_checkpoint(args.checkpoint, device=test_settings.training.device)
    model.load_state_dict(state["model_state_dict"])
    model.to(test_settings.training.device)
    model.eval()
    
    # 5. Run Inference
    logger.info("Running model inference on validation dataset...")
    all_images = []
    all_targets = []
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(test_settings.training.device)
            targets = batch["labels"]
            paths = batch["path"]
            
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_images.append(batch["image"])  # Kept on CPU
            all_targets.append(targets.numpy())
            all_probs.append(probs.cpu().numpy())
            all_paths.extend(paths)
            
    # Concatenate results
    all_images = torch.cat(all_images, dim=0)
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    
    class_names = val_loader.dataset.classes
    
    # 6. Compute Metrics
    logger.info("Computing validation metrics...")
    metrics = compute_metrics(all_targets, all_probs, class_names, threshold=args.threshold)
    
    # Save classification metrics report to JSON and text
    metrics_path_json = os.path.join(args.output_dir, "metrics_report.json")
    with open(metrics_path_json, "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Classification Report
    text_report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(text_report_path, "w") as f:
        f.write("=== ChestDiseaseAI Evaluation Report ===\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Macro AUROC: {metrics['macro_auroc']:.4f}\n")
        f.write(f"Macro F1-Score: {metrics['macro_f1']:.4f}\n")
        f.write(f"Macro Accuracy: {metrics['macro_accuracy']:.4f}\n\n")
        f.write("Per-Class Details:\n")
        for name in class_names:
            c = metrics["class_metrics"][name]
            au = f"{c['auroc']:.4f}" if c["auroc"] is not None else "N/A"
            f.write(f"{name}:\n")
            f.write(f"  AUROC:     {au}\n")
            f.write(f"  F1-Score:  {c['f1']:.4f}\n")
            f.write(f"  Precision: {c['precision']:.4f}\n")
            f.write(f"  Recall:    {c['recall']:.4f}\n")
            f.write(f"  Accuracy:  {c['accuracy']:.4f}\n")
            
    logger.info(f"Saved metric reports to: {args.output_dir}")
    
    # 7. Generate performance curves and confusion matrices
    logger.info("Generating evaluation charts (ROC, PR, Confusion Matrices)...")
    generate_curves_and_matrices(all_targets, all_probs, class_names, args.output_dir)
    
    # 8. Explainability: Initialize GradCAM Hook
    # We resolve the last conv layer dynamically from the model
    # For DenseNet121, this is model.model.features (last features block output)
    target_layer = model._get_classifier_head() # Or we hook features
    if test_settings.model.backbone.lower() == "densenet121":
        # Hook denseblock4 to avoid norm5 in-place ReLU errors
        target_layer = model.model.features.denseblock4
    elif test_settings.model.backbone.lower() == "efficientnet_b0":
        target_layer = model.model.features[-1]
    elif test_settings.model.backbone.lower() == "convnext_tiny":
        target_layer = model.model.features[-1]
    
    gradcam = GradCAM(model, target_layer)
    
    # 9. Error Triage (Triage False Positives and False Negatives)
    logger.info("Triaging errors and generating Grad-CAM overlays...")
    errors_list = []
    error_output_dir = os.path.join(args.output_dir, "errors")
    os.makedirs(error_output_dir, exist_ok=True)
    
    error_counter = 0
    # Loop over classes to pick errors
    for c_idx, name in enumerate(class_names):
        true_c = all_targets[:, c_idx]
        pred_c = all_probs[:, c_idx]
        
        valid = true_c != -1.0
        indices = np.arange(len(true_c))[valid]
        
        # False Positives: true is 0, predicted is high (closer to 1)
        fp_indices = indices[(true_c[valid] == 0.0) & (pred_c[valid] >= args.threshold)]
        if len(fp_indices) > 0:
            # Sort by highest predicted probability
            fp_indices = fp_indices[np.argsort(pred_c[fp_indices])[::-1]][:2]  # Keep top 2 FPs
            for idx in fp_indices:
                error_counter += 1
                overlay_path = os.path.join(error_output_dir, f"error_{error_counter:03d}_fp_{name}.png")
                
                # Generate Grad-CAM for this class on this image
                img_tensor = all_images[idx].unsqueeze(0).to(test_settings.training.device)
                heatmap = gradcam.generate_heatmap(img_tensor, c_idx)
                
                # Plot side by side prediction visualizer
                true_labels_list = [class_names[j] for j, val in enumerate(all_targets[idx]) if val == 1.0]
                true_labels_str = ", ".join(true_labels_list) if true_labels_list else "No Findings"
                
                fig = plot_prediction_with_gradcam(
                    image_tensor=all_images[idx],
                    true_labels=torch.tensor(all_targets[idx]),
                    pred_probs=all_probs[idx],
                    class_names=class_names,
                    gradcam_heatmaps={name: heatmap},
                    save_path=overlay_path
                )
                
                errors_list.append({
                    "error_type": "False Positive",
                    "target_class": name,
                    "image_path": all_paths[idx],
                    "overlay_path": overlay_path,
                    "true_labels_str": true_labels_str,
                    "pred_probs": all_probs[idx].tolist()
                })
                
        # False Negatives: true is 1, predicted is low (closer to 0)
        fn_indices = indices[(true_c[valid] == 1.0) & (pred_c[valid] < args.threshold)]
        if len(fn_indices) > 0:
            # Sort by lowest predicted probability
            fn_indices = fn_indices[np.argsort(pred_c[fn_indices])][:2]  # Keep top 2 FNs
            for idx in fn_indices:
                error_counter += 1
                overlay_path = os.path.join(error_output_dir, f"error_{error_counter:03d}_fn_{name}.png")
                
                # Generate Grad-CAM
                img_tensor = all_images[idx].unsqueeze(0).to(test_settings.training.device)
                heatmap = gradcam.generate_heatmap(img_tensor, c_idx)
                
                true_labels_list = [class_names[j] for j, val in enumerate(all_targets[idx]) if val == 1.0]
                true_labels_str = ", ".join(true_labels_list) if true_labels_list else "No Findings"
                
                fig = plot_prediction_with_gradcam(
                    image_tensor=all_images[idx],
                    true_labels=torch.tensor(all_targets[idx]),
                    pred_probs=all_probs[idx],
                    class_names=class_names,
                    gradcam_heatmaps={name: heatmap},
                    save_path=overlay_path
                )
                
                errors_list.append({
                    "error_type": "False Negatives",
                    "target_class": name,
                    "image_path": all_paths[idx],
                    "overlay_path": overlay_path,
                    "true_labels_str": true_labels_str,
                    "pred_probs": all_probs[idx].tolist()
                })
                
    # Clear hooks
    gradcam.remove_hooks()
    
    # 10. Generate the final HTML Dashboard
    generate_html_dashboard(metrics, class_names, errors_list, args.output_dir)
    logger.info("Evaluation complete! Dashboard generated successfully.")

if __name__ == "__main__":
    run_evaluation()
