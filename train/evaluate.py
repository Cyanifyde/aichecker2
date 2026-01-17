import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, required=True)
    args = parser.parse_args()

    scores = np.load(args.scores)
    print("mean score", scores.mean())


if __name__ == "__main__":
    main()
