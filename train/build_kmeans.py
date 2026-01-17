import argparse
import logging
import time
import numpy as np

from tokenization.kmeans_codebook import build_codebook
from train.logging_utils import configure_logging, format_duration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--clusters", type=int, default=2048)
    parser.add_argument("--log-level", type=str, default=None)
    args = parser.parse_args()

    configure_logging(args.log_level)
    log = logging.getLogger("train.build_kmeans")

    start = time.perf_counter()
    embeddings = np.load(args.embeddings)
    log.info("Loaded embeddings %s from %s", embeddings.shape, args.embeddings)
    codebook = build_codebook(embeddings, num_clusters=args.clusters)
    np.save(args.out, codebook.centroids)
    log.info("Saved codebook centroids %s to %s", codebook.centroids.shape, args.out)
    log.info("Done in %s", format_duration(time.perf_counter() - start))


if __name__ == "__main__":
    main()
