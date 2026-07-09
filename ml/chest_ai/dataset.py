from abc import ABC, abstractmethod
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional

class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all Chest X-ray dataset loaders.
    Ensures that any implementation provides a standard interface for training pipelines.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        pass
        
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            Dict containing:
                "image": torch.Tensor (transformed C x H x W image)
                "labels": torch.Tensor (label vector for multi-label classification)
                "path": str (absolute path to the image file)
        """
        pass
        
    @property
    @abstractmethod
    def classes(self) -> List[str]:
        """Returns the list of class names targeted by the model."""
        pass


class CheXpertDataset(BaseDataset):
    """
    Pluggable dataset loader for Stanford's CheXpert dataset.
    Supports customizable uncertainty policies and label subset selection.
    """
    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        target_labels: List[str],
        transforms: Optional[Any] = None,
        uncertainty_policy: str = "U-Zeros",
    ):
        """
        Args:
            csv_path: Path to CheXpert CSV file (relative to data_dir or absolute).
            data_dir: Directory containing the CheXpert images and CSV.
            target_labels: List of disease label names to classify.
            transforms: Albumentations transforms to apply.
            uncertainty_policy: Policy for handling -1.0 labels ("U-Zeros", "U-Ones", "U-Ignore").
        """
        self.data_dir = data_dir
        self.csv_path = csv_path if os.path.isabs(csv_path) else os.path.join(data_dir, csv_path)
        self.target_labels = target_labels
        self.transforms = transforms
        self.uncertainty_policy = uncertainty_policy
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CheXpert metadata CSV not found at: {self.csv_path}")
            
        self.df = pd.read_csv(self.csv_path)
        self._process_labels()
        
    def _process_labels(self):
        """
        Process uncertainty labels and NaNs.
        - Blanks/NaNs are treated as 0 (negative/unmentioned).
        - Uncertainty (-1.0) is handled depending on the chosen policy:
            * U-Zeros: -1.0 -> 0.0
            * U-Ones: -1.0 -> 1.0
            * U-Ignore: -1.0 -> -1.0 (requires masked loss in trainer)
        """
        # Fill NaNs with 0 (unmentioned is assumed negative)
        self.df[self.target_labels] = self.df[self.target_labels].fillna(0.0)
        
        # Apply uncertainty policy
        for label in self.target_labels:
            if self.uncertainty_policy == "U-Zeros":
                self.df[label] = self.df[label].replace(-1.0, 0.0)
            elif self.uncertainty_policy == "U-Ones":
                self.df[label] = self.df[label].replace(-1.0, 1.0)
            elif self.uncertainty_policy == "U-Ignore":
                # Keep -1.0 as is for custom loss masking
                self.df[label] = self.df[label].replace(-1.0, -1.0)
            else:
                raise ValueError(f"Unknown uncertainty policy: {self.uncertainty_policy}")
                
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # Resolve image path relative to data_dir
        rel_img_path = row["Path"]
        # Handle cases where "Path" might be absolute or prefixed differently
        if os.path.isabs(rel_img_path):
            img_path = rel_img_path
        else:
            img_path = os.path.join(self.data_dir, rel_img_path)
            
        # Read image
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at path: {img_path}")
            
        image = cv2.imread(img_path)
        if image is None:
            raise IOError(f"Failed to read image at: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply albumentations transforms
        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image_tensor = augmented["image"]
        else:
            # Fallback to simple tensor conversion
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
        # Get target labels
        labels = row[self.target_labels].values.astype(np.float32)
        labels_tensor = torch.from_numpy(labels)
        
        return {
            "image": image_tensor,
            "labels": labels_tensor,
            "path": img_path
        }
        
    @property
    def classes(self) -> List[str]:
        return self.target_labels
