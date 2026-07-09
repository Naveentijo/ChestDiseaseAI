import torch
import torch.nn as nn
import torchvision.models as models
from ml.chest_ai.logger import logger

class ChestClassifier(nn.Module):
    """
    Modular classifier for Chest X-ray disease detection.
    Supports DenseNet121, EfficientNet, ConvNeXt, and Vision Transformers (ViT)
    with pre-trained weights and customizable classification heads.
    """
    def __init__(
        self,
        backbone: str = "densenet121",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        
        logger.info(f"Initializing ChestClassifier with backbone: {self.backbone_name} (pretrained={pretrained})")
        
        # Build selected backbone
        if self.backbone_name == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.model = models.densenet121(weights=weights)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            
        elif self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            
        elif self.backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.model = models.convnext_tiny(weights=weights)
            in_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            
        elif self.backbone_name == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            self.model = models.vit_b_16(weights=weights)
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            
        else:
            supported = ["densenet121", "efficientnet_b0", "convnext_tiny", "vit_b_16"]
            raise ValueError(f"Backbone '{self.backbone_name}' not supported. Choose from {supported}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns logits of shape (batch_size, num_classes).
        """
        return self.model(x)

    def freeze_backbone(self):
        """Freezes all layers except for the classification head."""
        logger.info(f"Freezing backbone weights for fine-tuning...")
        # First, set require_grad = False for all parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Unfreeze head parameters
        head = self._get_classifier_head()
        for param in head.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreezes all parameters in the model."""
        logger.info(f"Unfreezing all model parameters...")
        for param in self.parameters():
            param.requires_grad = True
            
    def _get_classifier_head(self) -> nn.Module:
        """Helper to retrieve head submodule dynamically."""
        if self.backbone_name in ["densenet121", "efficientnet_b0"]:
            return self.model.classifier
        elif self.backbone_name == "convnext_tiny":
            return self.model.classifier[2]
        elif self.backbone_name == "vit_b_16":
            return self.model.heads.head
