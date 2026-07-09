import os
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union

def denormalize_image(
    image_tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    Inverts the standard normalization transform (mean/std subtraction) on an image tensor
    and returns a standard HWC numpy array of dtype uint8 suitable for plotting.
    
    Args:
        image_tensor: PyTorch tensor of shape (3, H, W).
        mean: List of channels normalization means.
        std: List of channels normalization standard deviations.
        
    Returns:
        Numpy array of shape (H, W, 3) in [0, 255] uint8.
    """
    # Clone and bring to CPU
    img = image_tensor.detach().cpu().clone()
    
    # Check dimensions
    if len(img.shape) == 4:
        img = img.squeeze(0)  # Squeeze batch dimension if present
        
    # Convert C x H x W to H x W x C
    img_np = img.permute(1, 2, 0).numpy()
    
    # Denormalize
    mean = np.array(mean)
    std = np.array(std)
    img_np = (img_np * std) + mean
    
    # Clip and convert to uint8
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return img_np

def overlay_heatmap(
    image: Union[torch.Tensor, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Blends a 2D float32 heatmap in [0, 1] onto a denormalized chest X-ray.
    
    Args:
        image: PyTorch tensor of shape (3, H, W) or numpy array of shape (H, W, 3) in [0, 255].
        heatmap: 2D numpy array of shape (H, W) in [0, 1].
        alpha: Heatmap blend factor (transparency).
        colormap: OpenCV colormap code.
        
    Returns:
        Blended RGB numpy array of shape (H, W, 3) in [0, 255] uint8.
    """
    if isinstance(image, torch.Tensor):
        img_np = denormalize_image(image)
    else:
        img_np = image.copy()
        
    # Convert heatmap to [0, 255] uint8
    heatmap_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    
    # Apply colormap (yields BGR image)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert colormap BGR to RGB
    color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match image dimensions if needed
    if color_heatmap_rgb.shape[:2] != img_np.shape[:2]:
        color_heatmap_rgb = cv2.resize(color_heatmap_rgb, (img_np.shape[1], img_np.shape[0]))
        
    # Blend images
    blended = cv2.addWeighted(img_np, 1.0 - alpha, color_heatmap_rgb, alpha, 0)
    return blended

def plot_sample(
    image_tensor: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots a single Chest X-ray image and overlays active and inactive label statuses.
    """
    img_np = denormalize_image(image_tensor)
    labels_np = labels.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    ax.axis("off")
    
    # Compile label text overlay
    overlay_text = []
    for name, val in zip(class_names, labels_np):
        if val == 1.0:
            status = "POSITIVE"
            color = "lime"
        elif val == -1.0:
            status = "UNCERTAIN"
            color = "orange"
        else:
            status = "NEGATIVE"
            color = "lightgray"
            
        overlay_text.append(f"{name}: {status}")
        
    # Add text box overlay
    text_content = "\n".join(overlay_text)
    props = dict(boxstyle="round", facecolor="black", alpha=0.7)
    
    ax.text(
        0.03, 0.97, text_content, transform=ax.transAxes,
        fontsize=9, color="white", verticalalignment="top", bbox=props,
        fontfamily="monospace"
    )
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        
    return fig

def plot_batch(
    batch: Dict[str, Any],
    class_names: List[str],
    max_samples: int = 8,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots a grid of samples from a DataLoader batch, displaying their active target states.
    """
    images = batch["image"]
    labels = batch["labels"]
    batch_size = images.shape[0]
    num_to_plot = min(batch_size, max_samples)
    
    cols = 4 if num_to_plot >= 4 else num_to_plot
    rows = int(np.ceil(num_to_plot / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    if num_to_plot == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
        
    for i in range(num_to_plot):
        img_np = denormalize_image(images[i])
        labels_np = labels[i].detach().cpu().numpy()
        
        axes[i].imshow(img_np)
        axes[i].axis("off")
        
        pos_targets = [class_names[idx] for idx, val in enumerate(labels_np) if val == 1.0]
        unc_targets = [class_names[idx] for idx, val in enumerate(labels_np) if val == -1.0]
        
        title_lines = []
        if pos_targets:
            title_lines.append(f"Pos: {', '.join(pos_targets)}")
        if unc_targets:
            title_lines.append(f"Unc: {', '.join(unc_targets)}")
        if not title_lines:
            title_lines.append("No Findings / Neg")
            
        axes[i].set_title("\n".join(title_lines), fontsize=9, color="black", fontweight="bold")
        
    for j in range(num_to_plot, len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        
    return fig

def plot_prediction_with_gradcam(
    image_tensor: torch.Tensor,
    true_labels: torch.Tensor,
    pred_probs: np.ndarray,
    class_names: List[str],
    gradcam_heatmaps: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots a side-by-side diagnosis visualizer:
    - Left: Original Chest X-ray image.
    - Right: Grad-CAM heatmap overlays for active/predicted diseases.
    
    Args:
        image_tensor: Normalized image tensor of shape (3, H, W).
        true_labels: Ground truth labels (num_classes,).
        pred_probs: Probability outputs from model (num_classes,).
        class_names: Target class labels names.
        gradcam_heatmaps: Dictionary mapping class name to its 2D heatmap.
        save_path: Optional path to save visual outputs.
    """
    img_np = denormalize_image(image_tensor)
    true_np = true_labels.detach().cpu().numpy()
    
    # Filter classes that have a heatmap generated
    plotted_classes = [c for c in class_names if c in gradcam_heatmaps]
    num_plots = 1 + len(plotted_classes)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 5, 5))
    if num_plots == 1:
        axes = np.array([axes])
        
    # 1. Plot original image on first subplot
    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title("Original Chest X-ray", fontsize=10, fontweight="bold")
    
    # Text summary overlay of ground truths
    gt_list = [class_names[idx] for idx, val in enumerate(true_np) if val == 1.0]
    gt_text = f"Ground Truth: {', '.join(gt_list)}" if gt_list else "Ground Truth: No Findings"
    axes[0].text(
        0.05, 0.05, gt_text, transform=axes[0].transAxes,
        fontsize=9, color="yellow", fontweight="bold",
        bbox=dict(facecolor="black", alpha=0.6, boxstyle="round")
    )
    
    # 2. Plot Grad-CAM overlays
    for idx, c_name in enumerate(plotted_classes):
        heatmap = gradcam_heatmaps[c_name]
        blended = overlay_heatmap(img_np, heatmap, alpha=0.45)
        
        c_idx = class_names.index(c_name)
        prob = pred_probs[c_idx]
        true_val = true_np[c_idx]
        
        ax_idx = idx + 1
        axes[ax_idx].imshow(blended)
        axes[ax_idx].axis("off")
        
        # Title summarizing prediction confidence vs truth
        status = "Pos" if true_val == 1.0 else ("Unc" if true_val == -1.0 else "Neg")
        axes[ax_idx].set_title(
            f"Grad-CAM: {c_name}\nPred: {prob:.1%} (Truth: {status})",
            fontsize=10, fontweight="bold"
        )
        
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        
    return fig
