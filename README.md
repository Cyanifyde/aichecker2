# AI Checker 2 - Dataset Preparation

This project expects raw image folders at the repository root:

```
ai/   # AI-generated or AI-traced images
real/ # Real/non-AI images
```

## Setup

Install Python deps:

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Normalize into the training format

Run the dataset preparation script to letterbox every image to 512x512, deduplicate
by the normalized PNG bytes, and write a manifest for training.

```
python -m data.prepare_dataset \
  --ai-dir ai \
  --real-dir real \
  --output-dir data/processed \
  --workers 0
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

## Start training

Training scripts live under `train/` and are meant to be run from the command line.

```bash
python -m train.train_classifier --epochs 3
```

### End-to-end training pipeline

Use the pipeline runner to execute all training stages from a single YAML config:

```bash
python -m train.pipeline --config configs/train_pipeline.yaml
```

Update `configs/train_pipeline.yaml` with your artifact paths (embeddings, scores, etc.).

## Run the Web UI (inference)

Start the FastAPI server:

```bash
python -m uvicorn webui.backend.app:app --host 127.0.0.1 --port 8000
```

Or use the config-driven launcher:

```bash
python run_webui.py --config configs/webui.yaml
```

Then open `http://127.0.0.1:8000/` in your browser.

## Notes

- The normalized PNGs are the only images used for training; keep originals in `ai/` and `real/`.
- Duplicates are detected across both classes and only the first occurrence is kept.
- Use `--workers N` and `--log-level INFO` to speed up and monitor large dataset runs.
- `--no-png-optimize` is faster, but changes output filenames (hashes) and increases file size.
