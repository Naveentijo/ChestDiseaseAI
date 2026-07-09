import os
import time
import shutil
import tempfile
from tqdm import tqdm
from ml.chest_ai.config import settings
from ml.chest_ai.utils import generate_synthetic_chexpert
from ml.chest_ai.dataloader import get_dataloaders
from ml.chest_ai.logger import logger

def run_benchmark(num_samples: int = 128, batch_size: int = 32, num_workers: int = 0):
    """
    Benchmarks the data loading pipeline throughput using a temporary synthetic dataset.
    
    Args:
        num_samples: Number of synthetic images to generate for the benchmark.
        batch_size: Batch size of the DataLoader.
        num_workers: Number of subprocesses to use for data loading.
    """
    print(f"=== Starting DataLoader Benchmark ===")
    print(f"Num samples: {num_samples} | Batch size: {batch_size} | Workers: {num_workers}")
    
    # Create temporary directory for dataset
    temp_dir = tempfile.mkdtemp()
    try:
        # Generate synthetic dataset
        print("Generating synthetic images for benchmark...")
        start_gen = time.time()
        generate_synthetic_chexpert(
            data_dir=temp_dir,
            num_train_samples=num_samples,
            num_valid_samples=10,
            image_size=settings.data.image_size
        )
        print(f"Synthetic dataset generation took {time.time() - start_gen:.2f} seconds.")
        
        # Override configuration directory
        settings.data.data_dir = temp_dir
        settings.training.batch_size = batch_size
        
        # Initialize loaders
        train_loader, _, _ = get_dataloaders(settings, num_workers=num_workers)
        
        # Warmup pass (load 1 batch)
        print("Warming up loader...")
        iterator = iter(train_loader)
        _ = next(iterator)
        
        # Benchmark iteration
        print("Iterating over dataset...")
        start_time = time.time()
        num_batches = 0
        total_samples = 0
        
        for batch in tqdm(train_loader, desc="Benchmarking batches"):
            images = batch["image"]
            total_samples += images.shape[0]
            num_batches += 1
            
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate stats
        throughput = total_samples / duration
        avg_batch_time = (duration / num_batches) * 1000  # ms
        
        print("\n=== Benchmark Results ===")
        print(f"Total time:       {duration:.4f} seconds")
        print(f"Total batches:    {num_batches}")
        print(f"Total samples:    {total_samples}")
        print(f"Avg Batch Time:   {avg_batch_time:.2f} ms")
        print(f"Throughput:       {throughput:.2f} samples/second")
        print("=========================")
        
    finally:
        # Cleanup temporary files
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Allow configuring parameters
    run_benchmark(num_samples=128, batch_size=32, num_workers=0)
