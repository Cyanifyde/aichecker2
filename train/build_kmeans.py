import argparse
import numpy as np

from tokenization.kmeans_codebook import build_codebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--clusters", type=int, default=2048)
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    codebook = build_codebook(embeddings, num_clusters=args.clusters)
    np.save(args.out, codebook.centroids)


if __name__ == "__main__":
    main()
