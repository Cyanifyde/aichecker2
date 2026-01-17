import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from train.logging_utils import configure_logging, format_duration


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


def build_steps(cfg: dict[str, Any], *, log_level_override: str | None = None) -> list[StepConfig]:
    steps_cfg = cfg.get("train_pipeline", {})
    steps: list[StepConfig] = []

    logging_cfg = steps_cfg.get("logging", {}) or {}
    default_log_level = str(log_level_override or logging_cfg.get("level", "INFO"))
    default_log_every = str(logging_cfg.get("log_every", 50))

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
                "--log-level",
                default_log_level,
                "--log-every",
                default_log_every,
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
                "--log-level",
                default_log_level,
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
                "--log-level",
                default_log_level,
                "--log-every",
                default_log_every,
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
            args=["-m", "train.evaluate", "--scores", str(scores_path), "--log-level", default_log_level],
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
                "--log-every",
                default_log_every,
                "--log-level",
                default_log_level,
            ],
            required_files=[non_ai_path],
        )
    )

    return steps


def run_step(
    step: StepConfig,
    *,
    step_index: int,
    step_total: int,
    total_start: float,
    durations: list[float],
    log_level: str | None,
) -> None:
    log = logging.getLogger("train.pipeline")
    for path in step.required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input for {step.name} not found: {path}. "
                "Update configs/train_pipeline.yaml to point at your artifacts."
            )

    cmd = [sys.executable, *step.args]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if log_level:
        env["AICHECKER_LOG_LEVEL"] = log_level
    else:
        env.setdefault("AICHECKER_LOG_LEVEL", "INFO")

    step_start = time.perf_counter()
    log.info("Step %d/%d starting: %s", step_index, step_total, step.name)
    log.info("Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    step_elapsed = time.perf_counter() - step_start
    durations.append(step_elapsed)

    total_elapsed = time.perf_counter() - total_start
    remaining_steps = step_total - step_index
    avg = (sum(durations) / len(durations)) if durations else 0.0
    eta = avg * remaining_steps if remaining_steps > 0 and avg > 0 else 0.0
    log.info(
        "Step %d/%d done: %s (step %s, total %s, est remaining %s)",
        step_index,
        step_total,
        step.name,
        format_duration(step_elapsed),
        format_duration(total_elapsed),
        format_duration(eta) if eta > 0 else "?:??",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train_pipeline.yaml"))
    parser.add_argument("--log-level", type=str, default=None, help="Overrides train_pipeline.logging.level")
    args = parser.parse_args()

    configure_logging(args.log_level)
    log = logging.getLogger("train.pipeline")

    cfg = load_config(args.config)
    steps = build_steps(cfg, log_level_override=args.log_level)
    if not steps:
        log.warning("No steps configured.")
        return

    enabled_steps = [s for s in steps if s.enabled]
    step_total = len(enabled_steps)
    if step_total == 0:
        log.warning("All steps are disabled.")
        return

    total_start = time.perf_counter()
    durations: list[float] = []

    idx = 0
    for step in steps:
        if not step.enabled:
            log.info("Skipping %s (disabled)", step.name)
            continue
        idx += 1
        run_step(
            step,
            step_index=idx,
            step_total=step_total,
            total_start=total_start,
            durations=durations,
            log_level=args.log_level,
        )

    total_elapsed = time.perf_counter() - total_start
    log.info("Pipeline complete in %s", format_duration(total_elapsed))


if __name__ == "__main__":
    main()
