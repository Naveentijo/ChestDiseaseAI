import os
import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

from ml.chest_ai.config import settings, Settings
from ml.chest_ai.model import ChestClassifier
from ml.chest_ai.transforms import get_valid_transforms
from ml.chest_ai.checkpoint import CheckpointManager
from ml.chest_ai.gradcam import GradCAM
from ml.chest_ai.visualization import overlay_heatmap, denormalize_image
from ml.chest_ai.logger import logger

class MLService:
    """
    MLService encapsulates all machine learning logic, including image decoding,
    validations, preprocessing, model loading/weight cache, inference passes,
    and Grad-CAM explainability overlays.
    """
    def __init__(self, checkpoint_path: Optional[str] = None):
        # Resolve checkpoint path, default to best_model if none specified
        if checkpoint_path is None:
            checkpoint_path = os.path.join(settings.training.checkpoint_dir, "best_model.pth")
            
        self.checkpoint_path = checkpoint_path
        self.device = settings.training.device
        self.class_names = settings.data.target_labels
        
        self.model: Optional[ChestClassifier] = None
        self.gradcam: Optional[GradCAM] = None
        self.transforms = get_valid_transforms(settings.data.image_size)
        
    def load_model(self):
        """Loads model weights into memory at startup."""
        if self.model is not None:
            return  # Already loaded
            
        logger.info(f"MLService: Loading model from checkpoint {self.checkpoint_path}...")
        
        if not os.path.exists(self.checkpoint_path):
            # Fallback if checkpoint doesn't exist (e.g. fresh environment or dev setup)
            logger.warning(f"MLService: Checkpoint not found at {self.checkpoint_path}. Instantiating raw un-trained model for API initialization.")
            self.model = ChestClassifier(
                backbone=settings.model.backbone,
                num_classes=len(self.class_names),
                pretrained=False
            )
        else:
            self.model = ChestClassifier(
                backbone=settings.model.backbone,
                num_classes=len(self.class_names),
                pretrained=False
            )
            manager = CheckpointManager(os.path.dirname(self.checkpoint_path))
            state = manager.load_checkpoint(self.checkpoint_path, device=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            
        self.model.to(self.device)
        self.model.eval()
        
        # Resolve target layer for GradCAM hooks
        target_layer = self.model._get_classifier_head()
        if settings.model.backbone.lower() == "densenet121":
            target_layer = self.model.model.features.denseblock4
        elif settings.model.backbone.lower() in ["efficientnet_b0", "convnext_tiny"]:
            target_layer = self.model.model.features[-1]
            
        self.gradcam = GradCAM(self.model, target_layer)
        logger.info("MLService: Model and Grad-CAM hooks successfully initialized.")
        
    def predict(self, image_bytes: bytes) -> Tuple[Dict[str, float], float, List[str]]:
        """
        Runs validation transforms and model forward pass to return multi-label predictions.
        
        Args:
            image_bytes: Raw binary file upload bytes.
            
        Returns:
            Tuple of (predictions_dict, max_confidence, list_of_detected_diseases)
        """
        if self.model is None:
            self.load_model()
            
        # 1. Decode image bytes using OpenCV
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file uploaded. Failed to decode image.")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Preprocess (Albumentations valid transforms)
        augmented = self.transforms(image=img_rgb)
        img_tensor = augmented["image"].unsqueeze(0).to(self.device)  # Add batch dimension -> [1, 3, 224, 224]
        
        # 3. Model forward pass
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
        # 4. Map probabilities to classes
        predictions = {}
        detected_diseases = []
        max_prob = 0.0
        
        for c_idx, name in enumerate(self.class_names):
            prob = float(probs[c_idx])
            predictions[name] = prob
            max_prob = max(max_prob, prob)
            
            # Use threshold 0.5 for detection
            if prob >= 0.5:
                detected_diseases.append(name)
                
        return predictions, max_prob, detected_diseases
        
    def generate_gradcam_overlay(self, image_bytes: bytes, target_class: str) -> bytes:
        """
        Generates a Grad-CAM overlay image for the specific target class.
        
        Args:
            image_bytes: Raw binary image upload.
            target_class: Class label name to map (e.g. 'Cardiomegaly').
            
        Returns:
            Binary file bytes representing the blended PNG image.
        """
        if self.model is None or self.gradcam is None:
            self.load_model()
            
        if target_class not in self.class_names:
            raise ValueError(f"Target class '{target_class}' not found. Supported: {self.class_names}")
            
        class_idx = self.class_names.index(target_class)
        
        # Decode image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image.")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        augmented = self.transforms(image=img_rgb)
        img_tensor = augmented["image"].unsqueeze(0).to(self.device)
        
        # Generate heatmap
        heatmap = self.gradcam.generate_heatmap(img_tensor, class_idx)
        
        # Overlay heatmap on denormalized image (or raw image, but we overlay on original image)
        # We overlay on img_rgb directly to match the original uploaded resolution
        blended = overlay_heatmap(img_rgb, heatmap, alpha=0.45)
        
        # Convert back to BGR for OpenCV encoding
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        
        # Encode blended image to PNG bytes
        success, encoded_img = cv2.imencode(".png", blended_bgr)
        if not success:
            raise RuntimeError("Failed to encode Grad-CAM image output.")
            
        return encoded_img.tobytes()
