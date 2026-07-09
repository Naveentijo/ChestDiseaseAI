import os
import glob
import re
import torch
from typing import Dict, Any, Optional, List
from ml.chest_ai.logger import logger

class CheckpointManager:
    """
    Manages saving and loading of model checkpoints, including training state preservation
    (optimizer, scheduler, epochs, metrics) and automatic file pruning.
    """
    def __init__(self, checkpoint_dir: str, max_to_keep: int = 3):
        """
        Args:
            checkpoint_dir: Directory where checkpoint files will be written.
            max_to_keep: Maximum number of epoch checkpoints to retain. Oldest will be pruned.
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        is_best: bool = False,
        epoch: int = 0
    ) -> str:
        """
        Saves the training state to a checkpoint file.
        
        Args:
            state: Dictionary containing:
                - 'model_state_dict': model weights
                - 'optimizer_state_dict': optimizer weights
                - 'scheduler_state_dict': scheduler weights (optional)
                - 'epoch': current epoch number
                - 'best_metric': best validation metric achieved
                - 'config': settings dictionary
            is_best: If True, saves an additional copy as 'best_model.pth'.
            epoch: The current training epoch.
            
        Returns:
            The file path of the saved checkpoint.
        """
        filename = f"checkpoint_epoch_{epoch:03d}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save checkpoint
        torch.save(state, filepath)
        logger.info(f"Saved training checkpoint to: {filepath}")
        
        # Handle best model saving
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(state, best_path)
            logger.info(f"New best model identified. Saved copy to: {best_path}")
            
        # Manage file pruning
        self._prune_checkpoints()
        
        return filepath
        
    def load_checkpoint(self, filepath: str, device: str = "cpu") -> Dict[str, Any]:
        """
        Loads training state from a checkpoint file.
        
        Args:
            filepath: Path to the checkpoint file (.pth).
            device: Target device to map tensors onto.
            
        Returns:
            The loaded state dictionary.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")
            
        logger.info(f"Loading checkpoint from: {filepath} onto device: {device}")
        state = torch.load(filepath, map_location=device)
        return state
        
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Scans the checkpoint directory and returns the path to the highest epoch checkpoint.
        
        Returns:
            Path to the latest checkpoint, or None if none exist.
        """
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth")
        checkpoints = glob.glob(pattern)
        
        if not checkpoints:
            return None
            
        # Parse epoch numbers from filenames
        epoch_checkpoints = []
        for path in checkpoints:
            match = re.search(r"checkpoint_epoch_(\d+)\.pth", os.path.basename(path))
            if match:
                epoch = int(match.group(1))
                epoch_checkpoints.append((epoch, path))
                
        if not epoch_checkpoints:
            return None
            
        # Return path of the highest epoch
        epoch_checkpoints.sort(key=lambda x: x[0])
        latest_path = epoch_checkpoints[-1][1]
        logger.info(f"Identified latest checkpoint file: {latest_path}")
        return latest_path
        
    def _prune_checkpoints(self):
        """Removes oldest epoch checkpoints if the count exceeds max_to_keep."""
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth")
        checkpoints = glob.glob(pattern)
        
        # Parse epoch numbers
        epoch_checkpoints = []
        for path in checkpoints:
            match = re.search(r"checkpoint_epoch_(\d+)\.pth", os.path.basename(path))
            if match:
                epoch = int(match.group(1))
                epoch_checkpoints.append((epoch, path))
                
        # If count exceeds limit, delete oldest
        if len(epoch_checkpoints) > self.max_to_keep:
            epoch_checkpoints.sort(key=lambda x: x[0])
            num_to_delete = len(epoch_checkpoints) - self.max_to_keep
            
            for i in range(num_to_delete):
                epoch, path = epoch_checkpoints[i]
                try:
                    os.remove(path)
                    logger.info(f"Pruned old checkpoint file: {path}")
                except OSError as e:
                    logger.warning(f"Failed to delete old checkpoint at {path}: {e}")
