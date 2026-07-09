import os
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class DataConfig(BaseModel):
    mode: Literal["local", "kaggle", "inference"] = "local"
    # Configurable data directory, defaults to environmental variable or local synthetic path
    data_dir: str = Field(default_factory=lambda: os.getenv("CHEXPERT_DATA_DIR", "./data/synthetic_chexpert"))
    
    # Paths relative to data_dir
    csv_train_path: str = "CheXpert-v1.0-small/train.csv"
    csv_valid_path: str = "CheXpert-v1.0-small/valid.csv"
    
    # Uncertainty handling policy: U-Zeros, U-Ones, or U-Ignore
    uncertainty_policy: Literal["U-Zeros", "U-Ones", "U-Ignore"] = "U-Zeros"
    
    # Label configuration
    use_competition_labels: bool = True
    
    competition_labels: List[str] = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion"
    ]
    
    all_labels: List[str] = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices"
    ]
    
    # Preprocessing
    image_size: int = 224

    @property
    def target_labels(self) -> List[str]:
        return self.competition_labels if self.use_competition_labels else self.all_labels


class ModelConfig(BaseModel):
    backbone: str = "densenet121"  # densenet121, efficientnet_b0, convnext_tiny, vit_b_16
    pretrained: bool = True
    dropout_rate: float = 0.2


class TrainingConfig(BaseModel):
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: Literal["Adam", "AdamW", "SGD"] = "AdamW"
    scheduler: Literal["ReduceLROnPlateau", "CosineAnnealingLR"] = "ReduceLROnPlateau"
    early_stopping_patience: int = 3
    mixed_precision: bool = True
    checkpoint_dir: str = "./ml/checkpoints"
    log_dir: str = "./ml/logs"
    device: str = "cpu"  # Will be dynamically set or overridden
    seed: int = 42


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHEST_AI_", case_sensitive=False)
    
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

# Global settings instance
settings = Settings()

# Automatically adjust device based on hardware if not explicitly set via env
if not os.getenv("CHEST_AI_TRAINING_DEVICE"):
    import torch
    if torch.cuda.is_available():
        settings.training.device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        settings.training.device = "mps"
    else:
        settings.training.device = "cpu"
