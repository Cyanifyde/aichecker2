import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import ImageFolderDataset
from models.graph_gnn import GraphClassifier
from models.token_transformer import TokenTransformer
from train.logging_utils import Progress, configure_logging, format_duration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    configure_logging(args.log_level)
    log = logging.getLogger("train.train_classifier")

    dataset = ImageFolderDataset([])
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    transformer = TokenTransformer()
    classifier = GraphClassifier()
    opt = torch.optim.Adam(list(transformer.parameters()) + list(classifier.parameters()), lr=1e-3)

    total_batches = len(loader) if hasattr(loader, "__len__") else 0
    progress = Progress(total=max(0, args.epochs * total_batches), log_every=max(1, int(args.log_every)), label="classifier")
    progress.start()

    if total_batches == 0:
        log.warning("Dataset is empty; no training batches will run.")

    for _ in range(args.epochs):
        for _imgs, _labels in loader:
            token_ids = torch.randint(0, 2048, (8, 128))
            coords = torch.rand(8, 128, 4)
            features = transformer(token_ids, coords)
            logits, ood = classifier(features)
            loss = ((logits - torch.rand_like(logits)) ** 2).mean() + ((ood) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            progress.update(1)
            if progress.should_log():
                msg, elapsed, _eta = progress.stats()
                log.info("%s | loss %.4f | elapsed %s", msg, float(loss.detach().cpu().item()), format_duration(elapsed))


if __name__ == "__main__":
    main()
