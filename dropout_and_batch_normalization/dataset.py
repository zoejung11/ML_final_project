import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CerebellumSliceDataset(Dataset):
    def __init__(self, data_root_path: str = "../data_processed", dataset_split: str = "train"):
        super().__init__()
        
        self.samples = []
        image_dir = Path(data_root_path) / "images" / dataset_split
        mask_dir = Path(data_root_path) / "masks" / dataset_split
        
        patients = os.listdir(image_dir)
        patients.sort()
        
        for patient in patients:
            patient_image_directory = image_dir / patient
            patient_mask_directory = mask_dir / patient
            slice_files = list(patient_image_directory.glob('*.png'))
            slice_files.sort()
                        
            for slice_file in slice_files:
                file_name = slice_file.name
                slice_path_str = str(patient_image_directory / file_name)
                mask_path_str = str(patient_mask_directory / file_name)
                self.samples.append((slice_path_str, mask_path_str, patient, file_name))
            
    def __len__(self):
        total_samples = len(self.samples)
        return total_samples
    
    def __getitem__(self, index):
        slice_path, mask_path, patient, filename = self.samples[index]
        
        raw_image_array = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
        float_image_array = raw_image_array.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(float_image_array).unsqueeze(0)
        
        raw_mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        float_mask_array = raw_mask_array.astype(np.float32) / 255.0
        
        mask_tensor = torch.from_numpy(float_mask_array).unsqueeze(0)
        
        binary_mask_tensor = (mask_tensor > 0.5).float()
        
        return image_tensor, binary_mask_tensor, patient, filename