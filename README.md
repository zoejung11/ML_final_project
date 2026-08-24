# ML Final Project: Cerebellum Segmentation Preprocessing

This repository prepares Mindboggle OASIS-TRT-20 structural MRI data and
trains cerebellum-segmentation models from the resulting PNG slices.

## What preprocessing produces

`Mindboggle/scripts/preprocess_mindboggle.py` converts each 3D MRI into 2D
coronal PNG slices and creates a matching binary cerebellum-mask PNG for each
slice. It also creates:

- `data_processed/splits.json`: deterministic subject-level train/validation/test split
- `data_processed/slices.csv`: one row per slice, including paths and whether
  the slice contains cerebellum

The script uses label IDs `6, 7, 45, 46, 630, 631, 632` as the cerebellum
target. These preprocessing choices should be recorded in any final report.

## Repository policy

Raw MRIs, generated processed data, model checkpoints, and virtual environments
are intentionally excluded from Git because they are large or reproducible.
Each collaborator creates their own environment and obtains the raw data locally.

## Setup

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Obtain the raw data

Download the OASIS-TRT-20 volumes and whole-brain volume-label archives from
the [Mindboggle data page](https://mindboggle.info/data). Place these two files
in `Mindboggle/data_raw/`:

```text
OASIS-TRT-20_volumes.tar.gz
WholeBrain_VolumeLabels_OASIS-TRT-20.tar.gz
```

Extract them into `Mindboggle/data_raw/extracted/`:

```bash
mkdir -p Mindboggle/data_raw/extracted
tar -xzf Mindboggle/data_raw/OASIS-TRT-20_volumes.tar.gz \
  -C Mindboggle/data_raw/extracted
tar -xzf Mindboggle/data_raw/WholeBrain_VolumeLabels_OASIS-TRT-20.tar.gz \
  -C Mindboggle/data_raw/extracted
```

After extraction, the required inputs are:

```text
Mindboggle/data_raw/extracted/
├── OASIS-TRT-20_volumes/OASIS-TRT-20-*/t1weighted_brain.nii.gz
└── OASIS-TRT-20_DKT31_CMA_labels_v2/
    └── OASIS-TRT-20-*_DKT31_CMA_labels.nii.gz
```

## Run preprocessing

Run this from the repository root:

```bash
.venv/bin/python Mindboggle/scripts/preprocess_mindboggle.py \
  --project-root Mindboggle --overwrite
```

`--overwrite` replaces the existing `Mindboggle/data_processed/` folder, so
use it only when you want to regenerate the complete processed dataset.

## Collaboration notes

- Commit changes to scripts, `Mindboggle/models/`, `requirements.txt`, and this README.
- Do not commit `.venv/`, raw data, processed data, or trained model files.
- If preprocessing choices change, rerun preprocessing and document the change
  before model training begins.

## Model training

`Mindboggle/models/` reads the already-generated contents of
`Mindboggle/data_processed/` and never modifies the preprocessing script or
any processed PNG. The fixed subject-level train/validation/test split created
by preprocessing is used as-is for every model.

### Basic ANN baseline

The first model is a pixel-wise artificial neural network. For each pixel, it
uses the preprocessed grayscale intensity and normalized x/y location to
predict whether that pixel belongs to the cerebellum. It is intentionally
non-convolutional, so it serves as a baseline against which CNNs and U-Nets
can demonstrate the benefit of spatial context.

From the repository root, train it with:

```bash
.venv/bin/python -m Mindboggle.models.train_ann --epochs 40 --batch-size 4
```

This produces local (Git-ignored) files in `Mindboggle/model_outputs/ann/`:

- `best_model.pt`: ANN weights selected by validation Dice.
- `metrics.json`: held-out test Dice, IoU, precision, recall, and specificity.

Use a separate command and output directory for each future model, but keep
the same processed data, subject splits, and evaluation protocol.

### Custom PyTorch 2D U-Net

`Mindboggle/models/unet.py` implements a three-level 2D U-Net using native
PyTorch layers. Its encoder learns progressively broader image features; its
decoder restores full-resolution predictions. Skip connections transfer fine
encoder details to the decoder to preserve cerebellar boundaries.

Train it from the repository root with the same data and split protocol as the
ANN:

```bash
.venv/bin/python -m Mindboggle.models.train_unet --epochs 40 --batch-size 4
```

The selected checkpoint and final test metrics are saved in
`Mindboggle/model_outputs/unet/`.

### MONAI 2D U-Net

`Mindboggle/models/monai_unet.py` configures MONAI's medical-imaging U-Net
with one grayscale input channel, one binary-mask output channel, feature
channels `(32, 64, 128, 256)`, three downsampling strides, and two residual
units per stage. It uses the same processed data, fixed subject split, loss,
validation selection, and final test metrics as the other models.

Train it from the repository root with:

```bash
.venv/bin/python -m Mindboggle.models.train_monai_unet --epochs 40 --batch-size 4
```

Its selected checkpoint and final test metrics are saved in
`Mindboggle/model_outputs/monai_unet/`.
