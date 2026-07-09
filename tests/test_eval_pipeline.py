import os
import shutil
import tempfile
import pytest
import matplotlib
matplotlib.use("Agg")
import torch
import numpy as np
import cv2

from ml.chest_ai.config import Settings
from ml.chest_ai.model import ChestClassifier
from ml.chest_ai.gradcam import GradCAM
from ml.chest_ai.visualization import overlay_heatmap, plot_prediction_with_gradcam, denormalize_image
from ml.chest_ai.evaluate import generate_curves_and_matrices, generate_html_dashboard

def test_gradcam_hook_and_generation():
    """Verifies that GradCAM hooks register, capture tensors, and generate valid heatmaps."""
    model = ChestClassifier(backbone="densenet121", num_classes=3, pretrained=False)
    
    # We hook the denseblock4 block inside features for DenseNet to avoid in-place errors
    target_layer = model.model.features.denseblock4
    gradcam = GradCAM(model, target_layer)
    
    # Single sample batch -> [1, 3, 224, 224]
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    
    # Generate heatmap for class 1
    heatmap = gradcam.generate_heatmap(x, class_idx=1)
    
    # Check shape and type
    assert heatmap.shape == (224, 224)
    assert heatmap.dtype == np.float32
    assert heatmap.max() <= 1.0
    assert heatmap.min() >= 0.0
    
    # Verify hooks captured non-none values
    assert gradcam.activations is not None
    assert gradcam.gradients is not None
    
    # Remove hooks
    gradcam.remove_hooks()


def test_overlay_heatmap():
    """Verifies overlay_heatmap blends images correctly."""
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_heatmap = np.random.rand(224, 224).astype(np.float32)
    
    blended = overlay_heatmap(dummy_img, dummy_heatmap, alpha=0.5)
    
    assert blended.shape == (224, 224, 3)
    assert blended.dtype == np.uint8


def test_prediction_with_gradcam_plotting():
    """Verifies plot_prediction_with_gradcam rendering and saving."""
    with tempfile.TemporaryDirectory() as temp_dir:
        img_tensor = torch.randn(3, 224, 224)
        true_labels = torch.tensor([1.0, 0.0, -1.0])
        pred_probs = np.array([0.85, 0.15, 0.45])
        class_names = ["Atelectasis", "Cardiomegaly", "Effusion"]
        
        heatmaps = {
            "Atelectasis": np.random.rand(224, 224).astype(np.float32),
            "Cardiomegaly": np.random.rand(224, 224).astype(np.float32)
        }
        
        save_path = os.path.join(temp_dir, "outputs", "pred_gradcam.png")
        fig = plot_prediction_with_gradcam(
            image_tensor=img_tensor,
            true_labels=true_labels,
            pred_probs=pred_probs,
            class_names=class_names,
            gradcam_heatmaps=heatmaps,
            save_path=save_path
        )
        
        assert os.path.exists(save_path)


def test_dashboard_and_curves_generators():
    """Verifies generation of evaluation plots and HTML dashboard."""
    with tempfile.TemporaryDirectory() as temp_dir:
        class_names = ["Atelectasis", "Cardiomegaly"]
        
        # 10 samples, 2 classes
        y_true = np.random.choice([0.0, 1.0, -1.0], size=(10, 2), p=[0.4, 0.4, 0.2])
        y_pred = np.random.rand(10, 2)
        
        # Call charts generator
        generate_curves_and_matrices(y_true, y_pred, class_names, temp_dir)
        
        assert os.path.exists(os.path.join(temp_dir, "roc_curves.png"))
        assert os.path.exists(os.path.join(temp_dir, "pr_curves.png"))
        assert os.path.exists(os.path.join(temp_dir, "confusion_matrices.png"))
        
        # Mock error list
        errors_list = [
            {
                "error_type": "False Positive",
                "target_class": "Cardiomegaly",
                "image_path": "/fake/path/img.jpg",
                "overlay_path": os.path.join(temp_dir, "errors", "err1.png"),
                "true_labels_str": "Atelectasis",
                "pred_probs": [0.1, 0.9]
            }
        ]
        
        # Create a mock error overlay image file so dashboard links are valid
        os.makedirs(os.path.join(temp_dir, "errors"), exist_ok=True)
        cv2.imwrite(os.path.join(temp_dir, "errors", "err1.png"), np.zeros((224, 224, 3), dtype=np.uint8))
        
        mock_metrics = {
            "macro_auroc": 0.85,
            "macro_f1": 0.75,
            "macro_precision": 0.70,
            "macro_recall": 0.80,
            "macro_accuracy": 0.78,
            "class_metrics": {
                "Atelectasis": {"auroc": 0.80, "f1": 0.72, "precision": 0.68, "recall": 0.76, "accuracy": 0.75},
                "Cardiomegaly": {"auroc": 0.90, "f1": 0.78, "precision": 0.72, "recall": 0.84, "accuracy": 0.81}
            }
        }
        
        # Call HTML generator
        generate_html_dashboard(mock_metrics, class_names, errors_list, temp_dir)
        
        assert os.path.exists(os.path.join(temp_dir, "dashboard.html"))
