import torch
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, CosineAnnealingLR
from typing import Union, Optional
from ml.chest_ai.config import Settings

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    settings: Settings
) -> Optional[Union[LRScheduler, ReduceLROnPlateau]]:
    """
    Factory function to get learning rate scheduler based on configurations.
    
    Args:
        optimizer: PyTorch optimizer instance.
        settings: Settings object.
        
    Returns:
        A PyTorch LR Scheduler or None.
    """
    scheduler_type = settings.training.scheduler
    
    if scheduler_type == "ReduceLROnPlateau":
        # Reduce LR when a metric has stopped improving
        return ReduceLROnPlateau(
            optimizer,
            mode="min",      # Monitor loss (minimize)
            factor=0.1,      # Reduce LR by 10x
            patience=2       # Wait 2 epochs
        )
    elif scheduler_type == "CosineAnnealingLR":
        # Cosine annealing decay
        return CosineAnnealingLR(
            optimizer,
            T_max=settings.training.epochs,
            eta_min=1e-6
        )
    else:
        return None
