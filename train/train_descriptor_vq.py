import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.datasets import ImageFolderDataset
from tokenization.vqgan_tokenizer import VQGANTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    dataset = ImageFolderDataset([])
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = VQGANTokenizer()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    for _ in range(args.epochs):
        for imgs, _ in loader:
            imgs = imgs.permute(0, 3, 1, 2)
            recon, vq = model(imgs)
            loss = loss_fn(recon, imgs) + vq.loss
            opt.zero_grad()
            loss.backward()
            opt.step()


if __name__ == "__main__":
    main()
