"""Shared, model-agnostic training and evaluation for the segmentation comparison."""

import csv
import json
import random
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from Mindboggle.models.data import CerebellumSliceDataset


def soft_dice_loss(logits, targets, epsilon=1e-6):
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (1 - (2 * intersection + epsilon) / (denominator + epsilon)).mean()


def calculate_metrics(logits, targets, threshold=0.5):
    """Return mean per-slice metrics, including positive-slice-only overlap."""
    prediction, truth = torch.sigmoid(logits) >= threshold, targets.bool()
    true_positive = (prediction & truth).sum(dim=(1, 2, 3)).float()
    false_positive = (prediction & ~truth).sum(dim=(1, 2, 3)).float()
    false_negative = (~prediction & truth).sum(dim=(1, 2, 3)).float()
    true_negative = (~prediction & ~truth).sum(dim=(1, 2, 3)).float()
    epsilon = 1e-6
    per_slice = {
        "dice": (2 * true_positive + epsilon) / (2 * true_positive + false_positive + false_negative + epsilon),
        "iou": (true_positive + epsilon) / (true_positive + false_positive + false_negative + epsilon),
        "precision": (true_positive + epsilon) / (true_positive + false_positive + epsilon),
        "recall": (true_positive + epsilon) / (true_positive + false_negative + epsilon),
        "specificity": (true_negative + epsilon) / (true_negative + false_positive + epsilon),
    }
    has_cerebellum = truth.sum(dim=(1, 2, 3)) > 0
    return {
        name: {"all_slices": value.mean().item(),
               "positive_slices": value[has_cerebellum].mean().item() if has_cerebellum.any() else None}
        for name, value in per_slice.items()
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits, masks = [], []
    for images, targets, _ in loader:
        logits.append(model(images.to(device)).cpu())
        masks.append(targets)
    return calculate_metrics(torch.cat(logits), torch.cat(masks))


@torch.no_grad()
def evaluate_epoch(model, loader, device, bce_loss):
    """Evaluate loss and Dice after an epoch without updating model weights."""
    model.eval()
    logits, masks, total_loss, count = [], [], 0.0, 0
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        output = model(images)
        batch_loss = bce_loss(output, targets) + soft_dice_loss(output, targets)
        total_loss += batch_loss.item() * images.size(0)
        count += images.size(0)
        logits.append(output.cpu())
        masks.append(targets.cpu())
    return total_loss / count, calculate_metrics(torch.cat(logits), torch.cat(masks))


@torch.no_grad()
def positive_slice_dice_totals(logits, targets, threshold=0.5, epsilon=1e-6):
    """Return Dice sum and slice count for non-empty masks in one batch."""
    prediction, truth = torch.sigmoid(logits) >= threshold, targets.bool()
    true_positive = (prediction & truth).sum(dim=(1, 2, 3)).float()
    false_positive = (prediction & ~truth).sum(dim=(1, 2, 3)).float()
    false_negative = (~prediction & truth).sum(dim=(1, 2, 3)).float()
    dice = (2 * true_positive + epsilon) / (2 * true_positive + false_positive + false_negative + epsilon)
    positive_slices = truth.sum(dim=(1, 2, 3)) > 0
    return dice[positive_slices].sum().item(), int(positive_slices.sum().item())


def next_run_suffix(output_dir):
    """Return '' for the first run, then '_2', '_3', ... without overwriting."""
    run_numbers = []
    for path in output_dir.glob("metrics*.json"):
        match = re.fullmatch(r"metrics(?:_(\d+))?\.json", path.name)
        if match:
            run_numbers.append(int(match.group(1) or 1))
    next_number = max(run_numbers, default=0) + 1
    return "" if next_number == 1 else f"_{next_number}"


def save_model_summary(model, data_root, output_path, device):
    """Save a Keras-style layer/output-shape/parameter-count CSV summary."""
    sample_image, _, _ = CerebellumSliceDataset(data_root, "train")[0]
    records, handles = [], []

    def shape_text(tensor):
        return "(" + ", ".join(["None", *map(str, tensor.shape[1:])]) + ")"

    for name, layer in model.named_modules():
        if name and not list(layer.children()):
            def capture(_, __, output, layer_name=name, layer_type=layer.__class__.__name__):
                if isinstance(output, torch.Tensor):
                    records.append({"layer": f"{layer_name} ({layer_type})", "output_shape": shape_text(output),
                                    "parameters": sum(parameter.numel() for parameter in _.parameters(recurse=False))})
            handles.append(layer.register_forward_hook(capture))
    model.eval()
    with torch.no_grad():
        output = model(sample_image.unsqueeze(0).to(device))
    for handle in handles:
        handle.remove()
    records.append({"layer": "model output (logits)", "output_shape": shape_text(output), "parameters": 0})
    records.append({"layer": "TOTAL TRAINABLE PARAMETERS", "output_shape": "", "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("layer", "output_shape", "parameters"))
        writer.writeheader()
        writer.writerows(records)
    return records


def train_and_test(model, model_name, architecture, data_root, output_dir, epochs, batch_size, learning_rate, seed):
    """Train on the fixed train split, select by validation Dice, test once."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set = CerebellumSliceDataset(data_root, "train")
    validation_set = CerebellumSliceDataset(data_root, "val")
    test_set = CerebellumSliceDataset(data_root, "test")
    print(f"Slices — train: {len(train_set)}, validation: {len(validation_set)}, test: {len(test_set)}")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_suffix = next_run_suffix(output_dir)
    model = model.to(device)
    save_model_summary(model, data_root, output_dir / f"model_summary{run_suffix}.csv", device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    bce_loss, best_validation_dice, best_state = nn.BCEWithLogitsLoss(), -1.0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        training_dice_sum, training_positive_slice_count = 0.0, 0
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = bce_loss(logits, targets) + soft_dice_loss(logits, targets)
            dice_sum, positive_count = positive_slice_dice_totals(logits.detach(), targets)
            training_dice_sum += dice_sum
            training_positive_slice_count += positive_count
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_set)
        train_positive_dice = training_dice_sum / training_positive_slice_count
        validation_loss, validation_metrics = evaluate_epoch(model, validation_loader, device, bce_loss)
        validation_dice = validation_metrics["dice"]["positive_slices"]
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                        "train_positive_slice_dice": train_positive_dice,
                        "validation_positive_slice_dice": validation_dice})
        print(f"Epoch {epoch}: train loss={train_loss:.4f}, validation loss={validation_loss:.4f}, "
              f"validation positive-slice Dice={validation_dice:.4f}")
        if validation_dice > best_validation_dice:
            best_validation_dice = validation_dice
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    results = {"model": model_name, "architecture": architecture,
               "best_validation_positive_slice_dice": best_validation_dice,
               "test_metrics": evaluate(model, test_loader, device)}
    torch.save(model.state_dict(), output_dir / f"best_model{run_suffix}.pt")
    with (output_dir / f"training_history{run_suffix}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (output_dir / f"metrics{run_suffix}.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
