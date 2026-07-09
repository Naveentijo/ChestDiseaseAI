import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_train_transforms(image_size: int) -> A.Compose:
    """
    Returns albumentations Compose object for training augmentations.
    Uses Standard ImageNet mean and std for transfer learning compatibility.
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_valid_transforms(image_size: int) -> A.Compose:
    """
    Returns albumentations Compose object for validation/inference normalization.
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
