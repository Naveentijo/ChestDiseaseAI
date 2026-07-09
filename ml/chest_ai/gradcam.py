import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional

class GradCAM:
    """
    Hook-based Grad-CAM (Gradient-weighted Class Activation Mapping) generator
    to compute explainability heatmaps showing model focus areas.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: PyTorch model instance (e.g. ChestClassifier).
            target_layer: Submodule layer to hook (usually the last convolutional layer).
        """
        self.model = model
        self.target_layer = target_layer
        
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        # Handle deprecation of register_backward_hook in newer PyTorch versions
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        self.activations = output
        
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generates a 2D Grad-CAM heatmap for a given class index.
        
        Args:
            input_tensor: Normalized image tensor of shape (1, 3, H, W).
            class_idx: Index of target class to visualize.
            
        Returns:
            Normalized 2D numpy array of shape (H, W) in [0, 1].
        """
        self.model.eval()
        
        # Ensure input tensor requires gradients
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.clone().requires_grad_(True)
            
        # Forward pass
        logits = self.model(input_tensor)
        
        # Get target score
        if class_idx < 0 or class_idx >= logits.shape[1]:
            raise ValueError(f"Invalid class index {class_idx}. Model has {logits.shape[1]} outputs.")
            
        score = logits[0, class_idx]
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Extract features and gradients from hooks
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks failed to capture activations or gradients. Ensure forward and backward runs occur.")
            
        # Convert to numpy arrays on CPU
        # Shape: (Channels, H_feat, W_feat)
        acts = self.activations.detach().cpu().numpy()[0]
        grads = self.gradients.detach().cpu().numpy()[0]
        
        # Global average pooling of gradients (channel weights alpha)
        weights = np.mean(grads, axis=(1, 2))
        
        # Compute weighted sum of activations
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
            
        # Apply ReLU to focus only on positive correlations
        cam = np.maximum(cam, 0)
        
        # Resize to input tensor image size (Width, Height)
        _, _, H, W = input_tensor.shape
        cam_resized = cv2.resize(cam, (W, H))
        
        # Normalize to [0, 1] range
        denom = cam_resized.max() - cam_resized.min()
        if denom == 0:
            return np.zeros_like(cam_resized)
            
        cam_normalized = (cam_resized - cam_resized.min()) / denom
        return cam_normalized
        
    def remove_hooks(self):
        """Removes registered hooks to prevent memory leaks."""
        self.forward_hook.remove()
        self.backward_hook.remove()
        
    def __del__(self):
        # Fallback to clear hooks on garbage collection
        try:
            self.remove_hooks()
        except Exception:
            pass
