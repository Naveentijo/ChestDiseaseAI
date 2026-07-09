import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import argparse
from ml.chest_ai.config import settings
from ml.chest_ai.utils import generate_synthetic_chexpert
from ml.chest_ai.dataloader import get_dataloaders
from ml.chest_ai.model import ChestClassifier
from ml.chest_ai.loss import get_loss_function
from ml.chest_ai.scheduler import get_scheduler
from ml.chest_ai.checkpoint import CheckpointManager
from ml.chest_ai.trainer import Trainer
from ml.chest_ai.logger import logger
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Train ChestDiseaseAI classifier.")
    parser.add_argument("--data-dir", type=str, help="Override root dataset directory.")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, help="Batch size.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--backbone", type=str, help="Model backbone (densenet121, efficientnet_b0, etc.).")
    parser.add_argument(
        "--uncertainty-policy",
        type=str,
        choices=["U-Zeros", "U-Ones", "U-Ignore"],
        help="Policy for uncertainty labels."
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Update settings with CLI arguments if provided
    if args.data_dir:
        settings.data.data_dir = args.data_dir
    if args.epochs:
        settings.training.epochs = args.epochs
    if args.batch_size:
        settings.training.batch_size = args.batch_size
    if args.lr:
        settings.training.learning_rate = args.lr
    if args.backbone:
        settings.model.backbone = args.backbone
    if args.uncertainty_policy:
        settings.data.uncertainty_policy = args.uncertainty_policy
        
    # 2. Local mode: check/generate synthetic dataset
    if settings.data.mode == "local" and not os.path.exists(settings.data.data_dir):
        logger.info(f"Synthetic dataset not found at {settings.data.data_dir}. Generating mock dataset...")
        generate_synthetic_chexpert(
            data_dir=settings.data.data_dir,
            num_train_samples=64,
            num_valid_samples=16,
            image_size=settings.data.image_size
        )
        
    # 3. Setup PyTorch reproducibility
    torch.manual_seed(settings.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.training.seed)
        
    # 4. Get DataLoaders and pos_weight
    logger.info("Setting up dataloaders...")
    train_loader, val_loader, pos_weights = get_dataloaders(settings, num_workers=0)
    
    # 5. Build Model
    model = ChestClassifier(
        backbone=settings.model.backbone,
        num_classes=len(settings.data.target_labels),
        pretrained=settings.model.pretrained,
        dropout_rate=settings.model.dropout_rate
    )
    
    # 6. Initialize Optimizer
    optimizer_type = settings.training.optimizer
    lr = settings.training.learning_rate
    wd = settings.training.weight_decay
    
    if optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        
    # 7. Initialize Scheduler
    scheduler = get_scheduler(optimizer, settings)
    
    # 8. Initialize Checkpoint Manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=settings.training.checkpoint_dir,
        max_to_keep=3
    )
    
    # 9. Get custom loss function
    loss_fn = get_loss_function(settings, pos_weight=pos_weights)
    
    # 10. Instantiate Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_manager=checkpoint_manager,
        settings=settings
    )
    
    # 11. Resume if requested
    if args.resume:
        trainer.resume_training(args.resume)
        
    # 12. Run training
    results = trainer.fit(train_loader, val_loader)
    logger.info(f"Training completed successfully! Best macro AUROC: {results['best_macro_auroc']:.4f}")

if __name__ == "__main__":
    main()
