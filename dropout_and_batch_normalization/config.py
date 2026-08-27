import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model import RegularizedSegmentationCNN
from dataset import CerebellumSliceDataset
from loss import CombinedLoss

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

regularized_cnn = RegularizedSegmentationCNN(input_channels=1, output_channels=1).to(device)

criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

optimizer = optim.Adam(regularized_cnn.parameters(), lr=1e-3, weight_decay=1e-5)

train_dataset = CerebellumSliceDataset(data_root_path="../data_processed", dataset_split="train")
val_dataset = CerebellumSliceDataset(data_root_path="../data_processed", dataset_split="val")

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=0)