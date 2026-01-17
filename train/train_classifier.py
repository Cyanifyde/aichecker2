import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import ImageFolderDataset
from models.graph_gnn import GraphClassifier
from models.token_transformer import TokenTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    dataset = ImageFolderDataset([])
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    transformer = TokenTransformer()
    classifier = GraphClassifier()
    opt = torch.optim.Adam(list(transformer.parameters()) + list(classifier.parameters()), lr=1e-3)

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


if __name__ == "__main__":
    main()
