import os
import cv2
import shutil
import tempfile
import pytest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from ml.chest_ai.config import settings, Settings
from ml.chest_ai.utils import generate_synthetic_chexpert
from ml.chest_ai.dataset import CheXpertDataset, BaseDataset
from ml.chest_ai.transforms import get_train_transforms, get_valid_transforms
from ml.chest_ai.dataloader import get_dataloaders, calculate_pos_weights
from ml.chest_ai.visualization import denormalize_image, plot_sample, plot_batch
from ml.chest_ai.checkpoint import CheckpointManager

@pytest.fixture(scope="module")
def temp_dataset_dir():
    """Creates a temporary directory with a synthetic CheXpert dataset."""
    temp_dir = tempfile.mkdtemp()
    # Generate 15 train and 5 valid synthetic samples
    generate_synthetic_chexpert(
        data_dir=temp_dir,
        num_train_samples=15,
        num_valid_samples=5,
        image_size=224
    )
    yield temp_dir
    # Cleanup after tests finish
    shutil.rmtree(temp_dir)


def test_synthetic_generator(temp_dataset_dir):
    """Verifies that the synthetic generator creates directories and files correctly."""
    chexpert_root = os.path.join(temp_dataset_dir, "CheXpert-v1.0-small")
    train_csv = os.path.join(chexpert_root, "train.csv")
    valid_csv = os.path.join(chexpert_root, "valid.csv")
    
    assert os.path.exists(chexpert_root)
    assert os.path.exists(train_csv)
    assert os.path.exists(valid_csv)
    
    # Check CSV contents
    df_train = pd.read_csv(train_csv)
    assert len(df_train) == 15
    assert "Path" in df_train.columns
    assert "Cardiomegaly" in df_train.columns
    
    # Check that image file actually exists
    sample_path = df_train.iloc[0]["Path"]
    assert os.path.exists(os.path.join(temp_dataset_dir, sample_path))


def test_configuration(temp_dataset_dir):
    """Verifies settings configuration loading and label selection."""
    assert len(settings.data.competition_labels) == 5
    assert settings.data.use_competition_labels is True
    assert settings.data.target_labels == settings.data.competition_labels
    
    # Test setting override
    custom_settings = Settings()
    custom_settings.data.use_competition_labels = False
    assert custom_settings.data.target_labels == custom_settings.data.all_labels


def test_transforms():
    """Verifies that transforms produce expected shape and type."""
    # Create random image representing raw CV2 image (HxWxC)
    dummy_img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    
    train_tf = get_train_transforms(224)
    valid_tf = get_valid_transforms(224)
    
    out_train = train_tf(image=dummy_img)["image"]
    out_valid = valid_tf(image=dummy_img)["image"]
    
    # Verify shape is C x H x W (3, 224, 224)
    assert out_train.shape == (3, 224, 224)
    assert out_valid.shape == (3, 224, 224)
    assert isinstance(out_train, torch.Tensor)
    assert out_train.dtype == torch.float32


def test_base_dataset_inheritance(temp_dataset_dir):
    """Verifies CheXpertDataset implements BaseDataset interface."""
    dataset = CheXpertDataset(
        csv_path=settings.data.csv_train_path,
        data_dir=temp_dataset_dir,
        target_labels=settings.data.target_labels,
        uncertainty_policy="U-Zeros"
    )
    assert isinstance(dataset, BaseDataset)


@pytest.mark.parametrize("policy,expected_uncertain_value", [
    ("U-Zeros", 0.0),
    ("U-Ones", 1.0),
    ("U-Ignore", -1.0)
])
def test_uncertainty_policies(temp_dataset_dir, policy, expected_uncertain_value):
    """Verifies that different uncertainty policies map -1.0 correctly."""
    # Write a small custom CSV containing -1.0, 1.0, 0.0, and NaN
    chexpert_root = os.path.join(temp_dataset_dir, "CheXpert-v1.0-small")
    custom_csv_path = os.path.join(chexpert_root, f"test_policy_{policy}.csv")
    
    # Create raw image and save it
    img_dir = os.path.join(chexpert_root, "train", "patient_test", "study1")
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, "view.jpg")
    cv2.imwrite(img_path, np.zeros((224, 224, 3), dtype=np.uint8))
    
    df = pd.DataFrame([{
        "Path": f"CheXpert-v1.0-small/train/patient_test/study1/view.jpg",
        "Cardiomegaly": -1.0,  # uncertain
        "Edema": 1.0,         # positive
        "Consolidation": 0.0,  # negative
        "Atelectasis": np.nan, # blank/NaN
        "Pleural Effusion": 0.0
    }])
    df.to_csv(custom_csv_path, index=False)
    
    target_labels = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
    
    dataset = CheXpertDataset(
        csv_path=custom_csv_path,
        data_dir=temp_dataset_dir,
        target_labels=target_labels,
        uncertainty_policy=policy
    )
    
    item = dataset[0]
    labels = item["labels"]
    
    # Check that Cardiomegaly (idx 0) uncertainty mapped correctly
    assert labels[0] == expected_uncertain_value
    # Check that positive mapped correctly
    assert labels[1] == 1.0
    # Check that negative mapped correctly
    assert labels[2] == 0.0
    # Check that NaN mapped to 0.0
    assert labels[3] == 0.0


def test_dataloader_loading(temp_dataset_dir):
    """Verifies dataloader loading, batch retrieval, and pos_weight calculation."""
    # Override settings for this test
    test_settings = Settings()
    test_settings.data.data_dir = temp_dataset_dir
    test_settings.training.batch_size = 4
    
    train_loader, val_loader, pos_weights = get_dataloaders(
        settings=test_settings,
        num_workers=0
    )
    
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    
    # Assert pos_weights shape matches number of target labels
    assert pos_weights.shape == (5,)
    assert (pos_weights >= 0).all()
    
    # Iterate a batch
    batch = next(iter(train_loader))
    assert "image" in batch
    assert "labels" in batch
    assert "path" in batch
    
    # Assert batch size is 4 (or less if end of dataset, but we generated 15, so first batch is 4)
    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["labels"].shape == (4, 5)


def test_visualization(temp_dataset_dir):
    """Verifies image denormalization and plotting utilities."""
    # Build a tiny dataset and retrieve one item
    dataset = CheXpertDataset(
        csv_path=settings.data.csv_train_path,
        data_dir=temp_dataset_dir,
        target_labels=settings.data.target_labels,
        uncertainty_policy="U-Zeros"
    )
    item = dataset[0]
    image = item["image"]
    labels = item["labels"]
    
    # Test denormalize
    img_np = denormalize_image(image)
    assert img_np.shape == (224, 224, 3)
    assert img_np.dtype == np.uint8
    
    # Test plot_sample
    save_sample_path = os.path.join(temp_dataset_dir, "outputs", "sample.png")
    fig = plot_sample(image, labels, dataset.classes, save_path=save_sample_path)
    assert os.path.exists(save_sample_path)
    
    # Test plot_batch
    batch = {
        "image": image.unsqueeze(0), # Add batch dim -> [1, 3, 224, 224]
        "labels": labels.unsqueeze(0) # Add batch dim -> [1, 5]
    }
    save_batch_path = os.path.join(temp_dataset_dir, "outputs", "batch.png")
    fig_batch = plot_batch(batch, dataset.classes, max_samples=1, save_path=save_batch_path)
    assert os.path.exists(save_batch_path)


def test_checkpoint_manager():
    """Verifies that the CheckpointManager saves, loads, and prunes checkpoints correctly."""
    with tempfile.TemporaryDirectory() as temp_chk_dir:
        manager = CheckpointManager(checkpoint_dir=temp_chk_dir, max_to_keep=2)
        
        # Define mock state states
        state_1 = {"epoch": 1, "best_metric": 0.85, "model_state_dict": {"weight": torch.tensor([1.0])}}
        state_2 = {"epoch": 2, "best_metric": 0.87, "model_state_dict": {"weight": torch.tensor([2.0])}}
        state_3 = {"epoch": 3, "best_metric": 0.89, "model_state_dict": {"weight": torch.tensor([3.0])}}
        
        # Save checkpoints
        p1 = manager.save_checkpoint(state_1, epoch=1)
        p2 = manager.save_checkpoint(state_2, epoch=2)
        
        # Verify both checkpoints exist
        assert os.path.exists(p1)
        assert os.path.exists(p2)
        
        # Save third checkpoint, should prune state_1 (oldest epoch)
        p3 = manager.save_checkpoint(state_3, epoch=3)
        assert os.path.exists(p3)
        assert os.path.exists(p2)
        assert not os.path.exists(p1)  # Pruned!
        
        # Verify latest checkpoint resolution
        latest = manager.get_latest_checkpoint()
        assert latest == p3
        
        # Verify checkpoint loading
        loaded_state = manager.load_checkpoint(latest, device="cpu")
        assert loaded_state["epoch"] == 3
        assert loaded_state["best_metric"] == 0.89
        assert loaded_state["model_state_dict"]["weight"] == torch.tensor([3.0])
        
        # Save best model copy
        p_best = manager.save_checkpoint(state_3, is_best=True, epoch=3)
        best_path = os.path.join(temp_chk_dir, "best_model.pth")
        assert os.path.exists(best_path)
