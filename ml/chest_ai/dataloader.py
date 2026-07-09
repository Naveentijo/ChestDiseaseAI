import os
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from ml.chest_ai.config import Settings
from ml.chest_ai.dataset import CheXpertDataset
from ml.chest_ai.transforms import get_train_transforms, get_valid_transforms

def calculate_pos_weights(csv_path: str, target_labels: list[str], uncertainty_policy: str) -> torch.Tensor:
    """
    Dynamically calculates class-wise positive weights (pos_weight) for BCEWithLogitsLoss
    to counter severe class imbalance.
    
    Formula: pos_weight_c = (N - P_c - U_c) / P_c
    Where:
      - N is total samples
      - P_c is positive count for class c
      - U_c is uncertain/ignored count for class c (if U-Ignore is used)
    """
    df = pd.read_csv(csv_path)
    num_classes = len(target_labels)
    pos_weights = []
    
    for label in target_labels:
        col = df[label].fillna(0.0)
        
        # Determine positive count
        if uncertainty_policy == "U-Ones":
            pos_count = ((col == 1.0) | (col == -1.0)).sum()
            neg_count = (col == 0.0).sum()
        elif uncertainty_policy == "U-Zeros":
            pos_count = (col == 1.0).sum()
            neg_count = ((col == 0.0) | (col == -1.0)).sum()
        else: # U-Ignore
            pos_count = (col == 1.0).sum()
            neg_count = (col == 0.0).sum()
            
        # Avoid division by zero
        if pos_count == 0:
            weight = 1.0
        else:
            weight = float(neg_count) / float(pos_count)
            
        pos_weights.append(weight)
        
    return torch.tensor(pos_weights, dtype=torch.float32)

def get_dataloaders(
    settings: Settings,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Factory function to construct PyTorch DataLoaders for train and validation.
    Returns:
        Tuple of (train_loader, val_loader, pos_weights)
    """
    data_dir = settings.data.data_dir
    csv_train = os.path.join(data_dir, settings.data.csv_train_path)
    csv_valid = os.path.join(data_dir, settings.data.csv_valid_path)
    
    # Check dataset mode
    if settings.data.mode == "inference":
        raise ValueError("Cannot load training/validation loaders in inference-only mode.")
        
    # Get transforms
    train_transforms = get_train_transforms(settings.data.image_size)
    valid_transforms = get_valid_transforms(settings.data.image_size)
    
    # Datasets
    train_dataset = CheXpertDataset(
        csv_path=settings.data.csv_train_path,
        data_dir=data_dir,
        target_labels=settings.data.target_labels,
        transforms=train_transforms,
        uncertainty_policy=settings.data.uncertainty_policy
    )
    
    val_dataset = CheXpertDataset(
        csv_path=settings.data.csv_valid_path,
        data_dir=data_dir,
        target_labels=settings.data.target_labels,
        transforms=valid_transforms,
        uncertainty_policy=settings.data.uncertainty_policy
    )
    
    # Calculate pos_weights for the loss function
    pos_weights = calculate_pos_weights(
        csv_path=csv_train,
        target_labels=settings.data.target_labels,
        uncertainty_policy=settings.data.uncertainty_policy
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, pos_weights
