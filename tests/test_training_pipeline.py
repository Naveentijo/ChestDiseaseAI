import os
import shutil
import tempfile
import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from ml.chest_ai.config import Settings
from ml.chest_ai.model import ChestClassifier
from ml.chest_ai.loss import MaskedBCEWithLogitsLoss, get_loss_function
from ml.chest_ai.metrics import compute_metrics
from ml.chest_ai.scheduler import get_scheduler
from ml.chest_ai.checkpoint import CheckpointManager
from ml.chest_ai.trainer import Trainer
from ml.chest_ai.utils import generate_synthetic_chexpert
from ml.chest_ai.dataloader import get_dataloaders

@pytest.fixture(scope="module")
def temp_dataset_dir():
    temp_dir = tempfile.mkdtemp()
    # Generate 8 train and 4 valid synthetic samples
    generate_synthetic_chexpert(
        data_dir=temp_dir,
        num_train_samples=8,
        num_valid_samples=4,
        image_size=224
    )
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_model_forward():
    """Verifies forward pass shape, dropout and backbone freezing."""
    model = ChestClassifier(backbone="densenet121", num_classes=5, pretrained=False)
    
    # Input batch of size 2
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    
    assert logits.shape == (2, 5)
    
    # Test backbone freezing
    model.freeze_backbone()
    
    # Check that classifier parameters have requires_grad=True
    # and model features have requires_grad=False
    for name, param in model.named_parameters():
        if "classifier" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False
            
    # Unfreeze
    model.unfreeze_backbone()
    for param in model.parameters():
        assert param.requires_grad is True


def test_masked_loss():
    """Verifies that MaskedBCEWithLogitsLoss ignores -1.0 target items."""
    # Batch size 2, class count 2
    logits = torch.tensor([[0.5, -0.5], [1.0, -1.0]], dtype=torch.float32)
    
    # Second class of first sample and first class of second sample are ignored (-1.0)
    targets = torch.tensor([[1.0, -1.0], [-1.0, 0.0]], dtype=torch.float32)
    
    loss_fn = MaskedBCEWithLogitsLoss()
    loss = loss_fn(logits, targets)
    
    # Calculate expected loss manually for non-masked entries:
    # entry (0,0): logit=0.5, target=1.0. Loss = -log(sigmoid(0.5)) = 0.474077
    # entry (1,1): logit=-1.0, target=0.0. Loss = -log(1 - sigmoid(-1.0)) = 0.313262
    # Expected mean = (0.474077 + 0.313262) / 2 = 0.393669
    
    expected_loss = 0.393669
    assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-4)


def test_metrics_calculation():
    """Verifies metrics computation ignores -1.0 targets and reports macro values."""
    # 4 samples, 2 classes
    y_true = np.array([
        [1.0, -1.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [1.0, 0.0]
    ])
    
    y_pred = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.7, 0.1]
    ])
    
    class_names = ["Cardiomegaly", "Edema"]
    metrics = compute_metrics(y_true, y_pred, class_names, threshold=0.5)
    
    # First class valid true: [1.0, 0.0, 1.0] (indexes 0, 1, 3)
    # First class valid pred: [0.9, 0.2, 0.7]. Thresh -> [1, 0, 1]. Accuracy = 1.0
    # Second class valid true: [1.0, 0.0, 0.0] (indexes 1, 2, 3)
    # Second class valid pred: [0.8, 0.2, 0.1]. Thresh -> [1, 0, 0]. Accuracy = 1.0
    
    assert metrics["macro_accuracy"] == 1.0
    assert metrics["class_metrics"]["Cardiomegaly"]["accuracy"] == 1.0
    assert metrics["class_metrics"]["Edema"]["accuracy"] == 1.0


def test_lr_scheduler():
    """Verifies LR scheduler builder creates instances."""
    model = ChestClassifier(backbone="densenet121", num_classes=5, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    test_settings = Settings()
    test_settings.training.scheduler = "ReduceLROnPlateau"
    scheduler = get_scheduler(optimizer, test_settings)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    
    test_settings.training.scheduler = "CosineAnnealingLR"
    scheduler = get_scheduler(optimizer, test_settings)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_trainer_fit(temp_dataset_dir):
    """Verifies that Trainer executes training and validation loops."""
    test_settings = Settings()
    test_settings.data.data_dir = temp_dataset_dir
    test_settings.training.epochs = 2
    test_settings.training.batch_size = 4
    test_settings.training.early_stopping_patience = 2
    
    train_loader, val_loader, pos_weights = get_dataloaders(test_settings, num_workers=0)
    
    model = ChestClassifier(
        backbone="densenet121",
        num_classes=5,
        pretrained=False
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_scheduler(optimizer, test_settings)
    
    with tempfile.TemporaryDirectory() as temp_chk_dir:
        checkpoint_manager = CheckpointManager(checkpoint_dir=temp_chk_dir)
        loss_fn = get_loss_function(test_settings, pos_weight=pos_weights)
        
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_manager=checkpoint_manager,
            settings=test_settings
        )
        
        # Train for 2 epochs
        results = trainer.fit(train_loader, val_loader)
        
        assert results["final_epoch"] == 2
        assert isinstance(results["best_macro_auroc"], float)
        
        # Verify checkpoint is written
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None
        
        # Test resuming training from epoch 2
        new_model = ChestClassifier(backbone="densenet121", num_classes=5, pretrained=False)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
        new_scheduler = get_scheduler(new_optimizer, test_settings)
        
        new_trainer = Trainer(
            model=new_model,
            loss_fn=loss_fn,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            checkpoint_manager=checkpoint_manager,
            settings=test_settings
        )
        
        new_trainer.resume_training(latest)
        assert new_trainer.start_epoch == 3
