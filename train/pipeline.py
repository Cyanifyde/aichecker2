import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StepConfig:
    name: str
    enabled: bool
    args: list[str]
    required_files: list[Path]


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def build_steps(cfg: dict[str, Any]) -> list[StepConfig]:
    steps_cfg = cfg.get("train_pipeline", {})
    steps: list[StepConfig] = []

    vq_cfg = steps_cfg.get("vq_train", {})
    steps.append(
        StepConfig(
            name="train_descriptor_vq",
            enabled=vq_cfg.get("enabled", True),
            args=[
                "-m",
                "train.train_descriptor_vq",
                "--data",
                str(vq_cfg.get("data_dir", "data/processed")),
                "--epochs",
                str(vq_cfg.get("epochs", 5)),
            ],
            required_files=[],
        )
    )

    kmeans_cfg = steps_cfg.get("kmeans", {})
    embeddings_path = Path(kmeans_cfg.get("embeddings", "artifacts/descriptor_embeddings.npy"))
    steps.append(
        StepConfig(
            name="build_kmeans",
            enabled=kmeans_cfg.get("enabled", True),
            args=[
                "-m",
                "train.build_kmeans",
                "--embeddings",
                str(embeddings_path),
                "--out",
                str(kmeans_cfg.get("out", "artifacts/codebook.npy")),
                "--clusters",
                str(kmeans_cfg.get("clusters", 2048)),
            ],
            required_files=[embeddings_path],
        )
    )

    cls_cfg = steps_cfg.get("classifier", {})
    steps.append(
        StepConfig(
            name="train_classifier",
            enabled=cls_cfg.get("enabled", True),
            args=[
                "-m",
                "train.train_classifier",
                "--epochs",
                str(cls_cfg.get("epochs", 3)),
            ],
            required_files=[],
        )
    )

    eval_cfg = steps_cfg.get("evaluate", {})
    scores_path = Path(eval_cfg.get("scores", "artifacts/val_scores.npy"))
    steps.append(
        StepConfig(
            name="evaluate",
            enabled=eval_cfg.get("enabled", True),
            args=["-m", "train.evaluate", "--scores", str(scores_path)],
            required_files=[scores_path],
        )
    )

    thresh_cfg = steps_cfg.get("tune_thresholds", {})
    non_ai_path = Path(thresh_cfg.get("non_ai_scores", "artifacts/non_ai_scores.npy"))
    steps.append(
        StepConfig(
            name="tune_thresholds",
            enabled=thresh_cfg.get("enabled", True),
            args=[
                "-m",
                "train.tune_thresholds",
                "--non-ai",
                str(non_ai_path),
                "--target-fpr",
                str(thresh_cfg.get("target_fpr", 0.001)),
                "--ci",
                str(thresh_cfg.get("ci", 0.95)),
            ],
            required_files=[non_ai_path],
        )
    )

    return steps


def run_step(step: StepConfig) -> None:
    for path in step.required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input for {step.name} not found: {path}. "
                "Update configs/train_pipeline.yaml to point at your artifacts."
            )
    cmd = [sys.executable, *step.args]
    print(f"\n==> Running {step.name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train_pipeline.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    steps = build_steps(cfg)
    if not steps:
        print("No steps configured.")
        return

    for step in steps:
        if not step.enabled:
            print(f"Skipping {step.name} (disabled)")
            continue
        run_step(step)


if __name__ == "__main__":
    main()
