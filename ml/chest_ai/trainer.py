import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional

from ml.chest_ai.config import Settings
from ml.chest_ai.logger import logger
from ml.chest_ai.metrics import compute_metrics
from ml.chest_ai.checkpoint import CheckpointManager

# Try importing TensorBoard SummaryWriter, fallback if not installed
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard is not installed. Experiment tracking will fallback to text log files only.")

class Trainer:
    """
    Orchestrates the multi-label training and validation loops, mixed precision,
    early stopping, gradient clipping, TensorBoard logging, and checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        settings: Optional[Settings] = None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        
        # Load configs
        from ml.chest_ai.config import settings as global_settings
        self.settings = settings or global_settings
        self.device = self.settings.training.device
        
        # Setup device
        self.model.to(self.device)
        
        # Setup Mixed Precision
        self.use_amp = self.settings.training.mixed_precision and self.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        if self.use_amp:
            logger.info("AMP (Mixed Precision) enabled for CUDA training.")
            
        # Setup TensorBoard
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            log_dir = os.path.join(self.settings.training.log_dir, "tensorboard_" + time.strftime("%Y%m%d-%H%M%S"))
            self.writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"Initialized TensorBoard writer at: {log_dir}")
            
        # Early Stopping attributes
        self.patience = self.settings.training.early_stopping_patience
        self.best_metric = -float("inf")  # We optimize for macro AUROC
        self.patience_counter = 0
        self.start_epoch = 1
        
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Runs the full training process for the model.
        """
        epochs = self.settings.training.epochs
        logger.info(f"Starting training on device: {self.device} for {epochs} epochs.")
        
        for epoch in range(self.start_epoch, epochs + 1):
            start_time = time.time()
            
            # 1. Train epoch
            train_loss = self._train_epoch(train_loader, epoch)
            
            # 2. Validation epoch
            val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Get macro metric
            val_macro_auroc = val_metrics["macro_auroc"]
            epoch_time = time.time() - start_time
            
            # Print epoch logs
            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Macro AUROC: {val_macro_auroc:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # 3. Step learning rate scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # 4. TensorBoard Logging
            if self.writer is not None:
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Val", val_loss, epoch)
                self.writer.add_scalar("Metrics/Val_Macro_AUROC", val_macro_auroc, epoch)
                self.writer.add_scalar("Metrics/Val_Macro_F1", val_metrics["macro_f1"], epoch)
                self.writer.add_scalar("LearningRate", current_lr, epoch)
                
                # Class-wise AUROC logging
                for name, metrics in val_metrics["class_metrics"].items():
                    if metrics["auroc"] is not None:
                        self.writer.add_scalar(f"Class_AUROC/{name}", metrics["auroc"], epoch)
                        
            # 5. Checkpointing & Best Model check
            is_best = val_macro_auroc > self.best_metric
            if is_best:
                logger.info(f"Val Macro AUROC improved from {self.best_metric:.4f} to {val_macro_auroc:.4f}.")
                self.best_metric = val_macro_auroc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.checkpoint_manager is not None:
                state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    "best_metric": self.best_metric,
                    "config": self.settings.model_dump()
                }
                self.checkpoint_manager.save_checkpoint(state, is_best=is_best, epoch=epoch)
                
            # 6. Early Stopping check
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs. No improvement for {self.patience} epochs.")
                break
                
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            
        return {
            "best_macro_auroc": self.best_metric,
            "final_epoch": epoch
        }
        
    def resume_training(self, checkpoint_path: str):
        """Loads a checkpoint and updates internal state to resume training."""
        if self.checkpoint_manager is None:
            raise ValueError("Cannot resume training without a CheckpointManager configured.")
            
        state = self.checkpoint_manager.load_checkpoint(checkpoint_path, device=self.device)
        
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler and state.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
            
        self.start_epoch = state["epoch"] + 1
        self.best_metric = state.get("best_metric", -float("inf"))
        logger.info(f"Resumed training state. Ready to start from epoch {self.start_epoch} (Best Macro AUROC: {self.best_metric:.4f})")

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Training Epoch {epoch}")
        for batch in pbar:
            images = batch["image"].to(self.device)
            targets = batch["labels"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP autocast
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)
                
            # Backward pass with Scaler
            self.scaler.scale(loss).backward()
            
            # Gradient clipping (unscale first)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Step scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            
        return total_loss / len(loader)

    def _validate_epoch(self, loader: DataLoader) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                images = batch["image"].to(self.device)
                targets = batch["labels"].to(self.device)
                
                # Sigmoid predictions
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)
                probs = torch.sigmoid(logits)
                
                total_loss += loss.item()
                
                # Save predictions
                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                
        # Stack predictions
        all_targets = np.vstack(all_targets)
        all_probs = np.vstack(all_probs)
        
        # Calculate metrics
        metrics = compute_metrics(all_targets, all_probs, loader.dataset.classes)
        avg_loss = total_loss / len(loader)
        
        return avg_loss, metrics
