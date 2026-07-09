import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from typing import Dict, List, Any
from ml.chest_ai.logger import logger

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes multi-label classification metrics (AUROC, F1, Precision, Recall, Accuracy)
    for each disease class and reports macro averages.
    
    Robustly handles U-Ignore policies by ignoring sample positions where y_true is -1.0.
    
    Args:
        y_true: Binary ground truth labels of shape (N, C) (may contain -1.0).
        y_pred: Predicted class probabilities of shape (N, C) in [0, 1].
        class_names: List of strings naming target classes.
        threshold: Classification probability threshold.
        
    Returns:
        Dictionary of macro and per-class metrics.
    """
    class_metrics = {}
    macro_aurocs = []
    macro_f1s = []
    macro_precisions = []
    macro_recalls = []
    macro_accuracies = []
    
    for c_idx, class_name in enumerate(class_names):
        true_c = y_true[:, c_idx]
        pred_c = y_pred[:, c_idx]
        
        # Filter out samples marked as -1.0 (ignored under U-Ignore policy)
        valid_mask = true_c != -1.0
        true_c_clean = true_c[valid_mask]
        pred_c_clean = pred_c[valid_mask]
        
        # Apply threshold to get binary predictions
        pred_c_binary = (pred_c_clean >= threshold).astype(np.float32)
        
        # Metrics defaults
        auroc = np.nan
        f1 = 0.0
        precision = 0.0
        recall = 0.0
        accuracy = 0.0
        
        # Compute AUROC only if both classes are present in ground truth
        unique_classes = np.unique(true_c_clean)
        if len(unique_classes) == 2:
            try:
                auroc = roc_auc_score(true_c_clean, pred_c_clean)
                macro_aurocs.append(auroc)
            except Exception as e:
                logger.warning(f"Error calculating AUROC for {class_name}: {e}")
        else:
            logger.debug(f"Skipping AUROC for {class_name} because only one class value exists in targets.")
            
        # Compute other metrics
        if len(true_c_clean) > 0:
            f1 = f1_score(true_c_clean, pred_c_binary, zero_division=0)
            precision = precision_score(true_c_clean, pred_c_binary, zero_division=0)
            recall = recall_score(true_c_clean, pred_c_binary, zero_division=0)
            accuracy = accuracy_score(true_c_clean, pred_c_binary)
            
            macro_f1s.append(f1)
            macro_precisions.append(precision)
            macro_recalls.append(recall)
            macro_accuracies.append(accuracy)
            
        class_metrics[class_name] = {
            "auroc": float(auroc) if not np.isnan(auroc) else None,
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": float(accuracy)
        }
        
    # Compute macro averages
    mean_auroc = np.mean(macro_aurocs) if macro_aurocs else 0.0
    mean_f1 = np.mean(macro_f1s) if macro_f1s else 0.0
    mean_precision = np.mean(macro_precisions) if macro_precisions else 0.0
    mean_recall = np.mean(macro_recalls) if macro_recalls else 0.0
    mean_accuracy = np.mean(macro_accuracies) if macro_accuracies else 0.0
    
    return {
        "macro_auroc": float(mean_auroc),
        "macro_f1": float(mean_f1),
        "macro_precision": float(mean_precision),
        "macro_recall": float(mean_recall),
        "macro_accuracy": float(mean_accuracy),
        "class_metrics": class_metrics
    }
