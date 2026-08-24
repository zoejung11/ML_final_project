# ML Final Project: Cerebellum Segmentation

This repository prepares Mindboggle OASIS-TRT-20 structural MRI data for cerebellar-segmentation

## Preprocessing output

`Mindboggle/scripts/preprocess_mindboggle.py` converts each 3D MRI into 2D
coronal PNG slices and creates a matching binary cerebellum-mask PNG for each
slice. It also creates:

- `data_processed/splits.json`: subject-level train/validation/test split
- `data_processed/slices.csv`: one row per slice, including paths and if
  the slice contains cerebellum

The script uses label IDs `6, 7, 45, 46, 630, 631, 632` as the anatomical cerebellum identifiers. 

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
use it when you want to rerun the complete processed dataset.

