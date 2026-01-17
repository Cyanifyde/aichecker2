import argparse
import logging
import time

import numpy as np

from train.logging_utils import configure_logging, format_duration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, required=True)
    parser.add_argument("--log-level", type=str, default=None)
    args = parser.parse_args()

    configure_logging(args.log_level)
    log = logging.getLogger("train.evaluate")
    start = time.perf_counter()

    scores = np.load(args.scores)
    log.info("Loaded scores %s from %s", scores.shape, args.scores)
    log.info("Mean score %.6f", float(scores.mean()))
    log.info("Done in %s", format_duration(time.perf_counter() - start))


if __name__ == "__main__":
    main()
