# AI Checker 2 - Dataset Preparation

This project expects raw image folders at the repository root:

```
ai/   # AI-generated or AI-traced images
real/ # Real/non-AI images
```

## Normalize into the training format

Run the dataset preparation script to letterbox every image to 512x512, deduplicate
by the normalized PNG bytes, and write a manifest for training.

```
python -m data.prepare_dataset \
  --ai-dir ai \
  --real-dir real \
  --output-dir data/processed
```

This creates:

```
data/processed/
  ai/            # normalized PNGs
  real/          # normalized PNGs
  manifest.csv   # columns: path, label, source_path
```

The labels in `manifest.csv` are:

- `1` = AI
- `0` = REAL (non-AI)

## Loading the prepared dataset

Use the manifest with the dataset helper:

```python
from pathlib import Path
from data.datasets import ImageFolderDataset, load_records_from_manifest

records = load_records_from_manifest(Path("data/processed/manifest.csv"))
dataset = ImageFolderDataset(records, training=True)
```

## Notes

- The normalized PNGs are the only images used for training; keep originals in `ai/` and `real/`.
- Duplicates are detected across both classes and only the first occurrence is kept.
