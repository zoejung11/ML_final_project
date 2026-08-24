"""Create uniquely named report tables and plots from a completed model run."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


DISPLAY_NAMES = {"dice": "Dice", "iou": "IoU", "precision": "Precision", "recall": "Recall"}
REPORT_METRICS = tuple(DISPLAY_NAMES)


def existing_or_numbered(directory, filename):
    """Keep prior artifacts: use name, then name_2, name_3, and so on."""
    path = directory / filename
    if not path.exists():
        return path
    stem, suffix, number = path.stem, path.suffix, 2
    while (directory / f"{stem}_{number}{suffix}").exists():
        number += 1
    return directory / f"{stem}_{number}{suffix}"


def read_history(path):
    with path.open(newline="") as file:
        return [{key: (int(value) if key == "epoch" else float(value)) for key, value in row.items()}
                for row in csv.DictReader(file)]


def write_csv(rows, output_path):
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_dice(history, output_path, model_name):
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(epochs, [row["train_positive_slice_dice"] for row in history], label="Training Dice")
    axis.plot(epochs, [row["validation_positive_slice_dice"] for row in history], label="Validation Dice")
    axis.set(title=f"{model_name}: Training and Validation Dice Over Epochs", xlabel="Epoch", ylabel="Dice score", ylim=(0, 1))
    axis.legend(); axis.grid(True, alpha=0.4)
    figure.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close(figure)


def plot_loss(history, output_path, model_name):
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(epochs, [row["train_loss"] for row in history], label="Training Loss")
    axis.plot(epochs, [row["validation_loss"] for row in history], label="Validation Loss")
    axis.set(title=f"{model_name}: Training and Validation Loss Over Epochs", xlabel="Epoch", ylabel="BCE + soft Dice loss")
    axis.legend(); axis.grid(True, alpha=0.4)
    figure.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close(figure)


def plot_validation_dice_progression(history, output_path, model_name):
    """Plot validation Dice alone for a clear, report-ready learning curve."""
    epochs = [row["epoch"] for row in history]
    validation_dice = [row["validation_positive_slice_dice"] for row in history]
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(epochs, validation_dice, color="forestgreen", linewidth=2, label="Validation Dice")
    axis.set(title=f"{model_name}: Validation Dice Score Progression", xlabel="Epoch", ylabel="Dice score")
    axis.legend(); axis.grid(True, alpha=0.4, linestyle="--")
    figure.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close(figure)


def plot_test_metrics(results, output_path):
    metrics = [metric for metric in REPORT_METRICS if metric in results["test_metrics"]]
    all_slices = [results["test_metrics"][metric]["all_slices"] for metric in metrics]
    positive_slices = [results["test_metrics"][metric]["positive_slices"] for metric in metrics]
    positions = list(range(len(metrics)))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar([position - 0.2 for position in positions], all_slices, width=0.4, label="All test slices")
    axis.bar([position + 0.2 for position in positions], positive_slices, width=0.4, label="Positive-mask test slices")
    axis.set(title=f"{results['model']}: Unseen Data Test Performance", ylabel="Score", ylim=(0, 1), xticks=positions,
             xticklabels=[DISPLAY_NAMES[metric] for metric in metrics])
    axis.legend(); axis.grid(axis="y", alpha=0.4)
    figure.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close(figure)


def plot_model_summary(summary_path, output_path, model_name):
    with summary_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    figure_height = max(3, 0.45 * (len(rows) + 2))
    figure, axis = plt.subplots(figsize=(13, figure_height))
    axis.axis("off")
    table = axis.table(cellText=[[row["layer"], row["output_shape"], f"{int(row['parameters']):,}"] for row in rows],
                       colLabels=["Layer (type)", "Output Shape", "Param #"], colWidths=[0.48, 0.32, 0.20], loc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)
    axis.set_title(f"{model_name}: Model Summary", pad=18, fontsize=16)
    figure.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--run-number", type=int, default=1, help="1 uses unsuffixed files; 2 uses *_2 files.")
    args = parser.parse_args()
    suffix = "" if args.run_number == 1 else f"_{args.run_number}"
    results_dir = args.results_dir
    with (results_dir / f"metrics{suffix}.json").open() as file:
        results = json.load(file)
    history = read_history(results_dir / f"training_history{suffix}.csv")

    metric_rows = [{"metric": "Best validation Dice (positive slices)", "value": results["best_validation_positive_slice_dice"]}]
    for metric in REPORT_METRICS:
        if metric not in results["test_metrics"]:
            continue
        values = results["test_metrics"][metric]
        metric_rows += [{"metric": f"Test {DISPLAY_NAMES[metric]} (all slices)", "value": values["all_slices"]},
                        {"metric": f"Test {DISPLAY_NAMES[metric]} (positive slices)", "value": values["positive_slices"]}]
    created = [existing_or_numbered(results_dir, "epoch_performance_table.csv"),
               existing_or_numbered(results_dir, "report_metrics_table.csv"),
               existing_or_numbered(results_dir, "test_metrics.png")]
    write_csv(history, created[0]); write_csv(metric_rows, created[1]); plot_test_metrics(results, created[2])

    summary_path = results_dir / f"model_summary{suffix}.csv"
    if summary_path.is_file():
        summary_output = existing_or_numbered(results_dir, "model_summary.png")
        plot_model_summary(summary_path, summary_output, results["model"])
        created.append(summary_output)
    required = {"train_positive_slice_dice", "validation_positive_slice_dice", "train_loss", "validation_loss"}
    if required.issubset(history[0]):
        dice_output = existing_or_numbered(results_dir, "training_validation_dice.png")
        loss_output = existing_or_numbered(results_dir, "training_validation_loss.png")
        validation_dice_output = existing_or_numbered(results_dir, "validation_dice_progression.png")
        plot_dice(history, dice_output, results["model"]); plot_loss(history, loss_output, results["model"])
        plot_validation_dice_progression(history, validation_dice_output, results["model"])
        created += [dice_output, loss_output, validation_dice_output]
    else:
        print("This older run lacks per-epoch training Dice and validation loss; rerun it to create those two plots.")
    print("Created:")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
