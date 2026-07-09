import torch
import torch.nn as nn
from typing import Optional
from ml.chest_ai.config import Settings

class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with Logits Loss supporting:
    1. Class-wise positive weights (pos_weight) to combat class imbalance.
    2. Dynamic sample-wise label masking to ignore uncertainty labels (-1.0 values under U-Ignore policy).
    """
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions of shape (B, C) before sigmoid.
            targets: Target labels of shape (B, C), where -1.0 signifies ignored samples.
            
        Returns:
            Scalar loss tensor.
        """
        # Create a binary mask where target is not -1.0
        mask = (targets != -1.0).float()
        
        # Replace -1.0 with 0.0 in clean target. The actual value does not matter
        # since the mask will zero out its loss contribution.
        clean_targets = torch.where(targets == -1.0, torch.zeros_like(targets), targets)
        
        # Compute element-wise BCE with logits
        # If pos_weight is provided, pass it. We use reduction="none" to apply the mask manually.
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="none")
        element_loss = loss_fn(logits, clean_targets)
        
        # Apply the mask
        masked_loss = element_loss * mask
        
        # Average over active elements (avoiding division by zero)
        num_active = mask.sum()
        if num_active == 0:
            return masked_loss.sum()  # Will be 0
            
        return masked_loss.sum() / num_active

def get_loss_function(settings: Settings, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Factory function to get the loss function based on the configuration.
    
    If the uncertainty policy is U-Ignore, returns MaskedBCEWithLogitsLoss.
    Otherwise, returns standard nn.BCEWithLogitsLoss with pos_weight.
    """
    if settings.data.uncertainty_policy == "U-Ignore":
        return MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        # Standard BCEWithLogitsLoss handles U-Zeros and U-Ones as targets will already be mapped to 0/1
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
