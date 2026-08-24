#!/usr/bin/env python3
"""Create aligned coronal T1 images and binary whole-cerebellum masks.

Raw Mindboggle files are read only.  Outputs are written to data_processed/.
"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


TARGET_LABEL_IDS = (6, 7, 45, 46, 630, 631, 632)


def subject_splits(subjects, seed):
    """Make a deterministic 14/3/3 split of the 20 supplied volumes."""
    shuffled = subjects[:]
    random.Random(seed).shuffle(shuffled)
    return {"train": shuffled[:14], "val": shuffled[14:17], "test": shuffled[17:]}


def normalize_t1(image):
    """Robustly map positive brain voxels to an 8-bit grayscale image."""
    foreground = image[image > 0]
    if foreground.size == 0:
        raise ValueError("T1 image contains no positive voxels")
    lower, upper = np.percentile(foreground, (1, 99))
    if upper <= lower:
        raise ValueError("T1 intensity percentiles are invalid")
    return (255 * np.clip((image - lower) / (upper - lower), 0, 1)).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/project"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    source = project / "data_raw" / "extracted"
    output = project / "data_processed"
    volumes_root = source / "OASIS-TRT-20_volumes"
    labels_root = source / "OASIS-TRT-20_DKT31_CMA_labels_v2"

    subjects = [f"OASIS-TRT-20-{number}" for number in range(1, 21)]
    missing = [
        subject
        for subject in subjects
        if not (volumes_root / subject / "t1weighted_brain.nii.gz").is_file()
        or not (labels_root / f"{subject}_DKT31_CMA_labels.nii.gz").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing MRI or label file(s): {', '.join(missing)}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; re-run with --overwrite to replace it")
        shutil.rmtree(output)

    splits = subject_splits(subjects, args.seed)
    output.mkdir(parents=True)
    (output / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")

    manifest_rows = []
    for split, split_subjects in splits.items():
        image_dir = output / "images" / split
        mask_dir = output / "masks" / split
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        for subject in split_subjects:
            t1 = nib.as_closest_canonical(
                nib.load(str(volumes_root / subject / "t1weighted_brain.nii.gz"))
            )
            labels = nib.as_closest_canonical(
                nib.load(str(labels_root / f"{subject}_DKT31_CMA_labels.nii.gz"))
            )
            if t1.shape != labels.shape or not np.allclose(t1.affine, labels.affine):
                raise ValueError(f"Canonical MRI and label volume do not align for {subject}")

            image = normalize_t1(t1.get_fdata(dtype=np.float32))
            binary_mask = np.isin(np.asanyarray(labels.dataobj), TARGET_LABEL_IDS).astype(np.uint8)

            # In canonical RAS orientation, axis 1 indexes coronal planes.
            for coronal_index in range(image.shape[1]):
                # Rotate both arrays together solely for conventional screen display.
                image_slice = np.rot90(image[:, coronal_index, :])
                mask_slice = np.rot90(binary_mask[:, coronal_index, :])
                filename = f"{subject}_coronal-{coronal_index:03d}.png"
                Image.fromarray(image_slice, mode="L").save(image_dir / filename)
                Image.fromarray(mask_slice * 255, mode="L").save(mask_dir / filename)
                manifest_rows.append(
                    {
                        "subject": subject,
                        "split": split,
                        "coronal_index": coronal_index,
                        "image": str(Path("images") / split / filename),
                        "mask": str(Path("masks") / split / filename),
                        "cerebellum_pixels": int(mask_slice.sum()),
                        "has_cerebellum": int(mask_slice.any()),
                    }
                )

    with (output / "slices.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    totals = {split: sum(row["split"] == split for row in manifest_rows) for split in splits}
    positives = {
        split: sum(row["split"] == split and row["has_cerebellum"] for row in manifest_rows)
        for split in splits
    }
    print("Created", output)
    print("Slices per split:", totals)
    print("Slices containing cerebellum:", positives)
    print("Target label IDs:", TARGET_LABEL_IDS)


if __name__ == "__main__":
    main()
