import argparse
import logging
import time
from dataclasses import dataclass

import numpy as np

from train.logging_utils import Progress, configure_logging, format_duration


@dataclass
class ThresholdResult:
    tau_ai: float
    tau_unknown: float
    upper_fpr: float


def bootstrap_fpr(
    scores: np.ndarray,
    thresholds: np.ndarray,
    *,
    n_boot: int = 2000,
    log_every: int = 100,
    log: logging.Logger | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(7)
    fprs = np.zeros((n_boot, thresholds.shape[0]))
    progress = Progress(total=n_boot, log_every=max(1, int(log_every)), label="bootstrap") if log else None
    if progress:
        progress.start()
    for i in range(n_boot):
        sample = rng.choice(scores, size=len(scores), replace=True)
        for j, tau in enumerate(thresholds):
            fprs[i, j] = (sample >= tau).mean()
        if progress:
            progress.update(1)
            if progress.should_log():
                msg, elapsed, _eta = progress.stats()
                log.info("%s | elapsed %s", msg, format_duration(elapsed))
    return fprs


def tune_thresholds(
    non_ai_scores: np.ndarray,
    target_fpr: float,
    ci: float,
    *,
    n_boot: int = 2000,
    log_every: int = 100,
    log: logging.Logger | None = None,
) -> ThresholdResult:
    thresholds = np.linspace(0.5, 0.999, 200)
    fprs = bootstrap_fpr(non_ai_scores, thresholds, n_boot=n_boot, log_every=log_every, log=log)
    upper = np.quantile(fprs, ci, axis=0)
    idx = np.where(upper <= target_fpr)[0]
    tau_ai = float(thresholds[idx[0]]) if len(idx) else float(thresholds[-1])
    return ThresholdResult(tau_ai=tau_ai, tau_unknown=0.55, upper_fpr=float(upper[idx[0]] if len(idx) else upper[-1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--non-ai", type=str, required=True)
    parser.add_argument("--target-fpr", type=float, default=0.001)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    configure_logging(args.log_level)
    log = logging.getLogger("train.tune_thresholds")
    start = time.perf_counter()

    non_ai_scores = np.load(args.non_ai)
    log.info("Loaded non-ai scores %s from %s", non_ai_scores.shape, args.non_ai)
    result = tune_thresholds(
        non_ai_scores,
        args.target_fpr,
        args.ci,
        n_boot=args.n_boot,
        log_every=args.log_every,
        log=log,
    )
    log.info("Result: %s", result)
    log.info("Done in %s", format_duration(time.perf_counter() - start))


if __name__ == "__main__":
    main()
