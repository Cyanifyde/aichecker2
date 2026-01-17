import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class ThresholdResult:
    tau_ai: float
    tau_unknown: float
    upper_fpr: float


def bootstrap_fpr(scores: np.ndarray, thresholds: np.ndarray, n_boot: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(7)
    fprs = np.zeros((n_boot, thresholds.shape[0]))
    for i in range(n_boot):
        sample = rng.choice(scores, size=len(scores), replace=True)
        for j, tau in enumerate(thresholds):
            fprs[i, j] = (sample >= tau).mean()
    return fprs


def tune_thresholds(non_ai_scores: np.ndarray, target_fpr: float, ci: float) -> ThresholdResult:
    thresholds = np.linspace(0.5, 0.999, 200)
    fprs = bootstrap_fpr(non_ai_scores, thresholds)
    upper = np.quantile(fprs, ci, axis=0)
    idx = np.where(upper <= target_fpr)[0]
    tau_ai = float(thresholds[idx[0]]) if len(idx) else float(thresholds[-1])
    return ThresholdResult(tau_ai=tau_ai, tau_unknown=0.55, upper_fpr=float(upper[idx[0]] if len(idx) else upper[-1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--non-ai", type=str, required=True)
    parser.add_argument("--target-fpr", type=float, default=0.001)
    parser.add_argument("--ci", type=float, default=0.95)
    args = parser.parse_args()

    non_ai_scores = np.load(args.non_ai)
    result = tune_thresholds(non_ai_scores, args.target_fpr, args.ci)
    print(result)


if __name__ == "__main__":
    main()
