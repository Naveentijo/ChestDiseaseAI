import os
import cv2
import numpy as np
import pandas as pd
from typing import List

def generate_synthetic_chexpert(
    data_dir: str,
    num_train_samples: int = 10,
    num_valid_samples: int = 5,
    image_size: int = 224
):
    """
    Generates a synthetic CheXpert dataset directory mimicking the official Stanford structure.
    Creates mock images and corresponding train.csv/valid.csv metadata.
    
    Args:
        data_dir: Directory where the synthetic dataset will be created.
        num_train_samples: Number of training samples to generate.
        num_valid_samples: Number of validation samples to generate.
        image_size: Size of mock square images to generate.
    """
    labels = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
        "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
    ]
    
    # Establish subdirectories
    chexpert_root = os.path.join(data_dir, "CheXpert-v1.0-small")
    for split in ["train", "valid"]:
        split_dir = os.path.join(chexpert_root, split)
        os.makedirs(split_dir, exist_ok=True)
        
        num_samples = num_train_samples if split == "train" else num_valid_samples
        records = []
        
        for i in range(1, num_samples + 1):
            # Follow official hierarchical path structure
            patient_id = f"patient{i:05d}"
            study_id = "study1"
            view_name = "view1_frontal.jpg"
            
            patient_dir = os.path.join(split_dir, patient_id, study_id)
            os.makedirs(patient_dir, exist_ok=True)
            
            # Save synthetic image (grayscale gradient/noise mapped to 3 channels)
            img_filename = view_name
            img_path_full = os.path.join(patient_dir, img_filename)
            
            # Create a simple synthetic chest X-ray mock (gradient + circle representing heart/lungs)
            mock_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            cv2.circle(mock_img, (image_size // 2, image_size // 2), image_size // 4, (100, 100, 100), -1)
            cv2.GaussianBlur(mock_img, (21, 21), 0, mock_img)
            # Add some random noise
            noise = np.random.normal(0, 10, mock_img.shape).astype(np.int16)
            mock_img = np.clip(mock_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            cv2.imwrite(img_path_full, mock_img)
            
            # Relative path to match official CheXpert CSV style
            # CheXpert paths look like: CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
            relative_path = f"CheXpert-v1.0-small/{split}/{patient_id}/{study_id}/{view_name}"
            
            # Build mock record
            record = {
                "Path": relative_path,
                "Sex": np.random.choice(["Male", "Female"]),
                "Age": np.random.randint(18, 90),
                "Frontal/Lateral": "Frontal",
                "AP/PA": np.random.choice(["AP", "PA"]),
            }
            
            # Fill label values: 1.0 (pos), 0.0 (neg), -1.0 (uncertain), or NaN (empty/missing)
            for label in labels:
                val = np.random.choice([1.0, 0.0, -1.0, np.nan], p=[0.4, 0.4, 0.15, 0.05])
                record[label] = val
                
            records.append(record)
            
        # Write split metadata CSV
        df = pd.DataFrame(records)
        csv_filename = f"{split}.csv"
        csv_path = os.path.join(chexpert_root, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Generated synthetic metadata CSV at: {csv_path}")
