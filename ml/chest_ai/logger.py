import os
import logging
from logging.handlers import RotatingFileHandler
from ml.chest_ai.config import settings

def setup_logger(name: str = "chest_ai") -> logging.Logger:
    """
    Sets up a logger with dual outputs: stdout (console) and a rotating file.
    
    Args:
        name: Name of the logger instance.
        
    Returns:
        logging.Logger instance configured.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = settings.training.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "chest_ai.log")
    
    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # File Handler (Rotating: max 10MB per file, keeping up to 5 files)
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    except IOError as e:
        # Fallback if file logging cannot be established (e.g. read-only permissions)
        console_handler.warning(f"Failed to initialize rotating file log at {log_file} due to: {e}")
        
    return logger

# Common global logger
logger = setup_logger("chest_ai")
