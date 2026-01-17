import argparse
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.datasets import ImageFolderDataset, load_records_from_manifest
from tokenization.vqgan_tokenizer import VQGANTokenizer
from train.logging_utils import Progress, configure_logging, format_duration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    configure_logging(args.log_level)
    log = logging.getLogger("train.train_descriptor_vq")

    data_path = Path(args.data)
    manifest_path = data_path if data_path.suffix.lower() == ".csv" else data_path / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "Run `python -m data.prepare_dataset --output-dir data/processed` or pass a manifest.csv path."
        )

    records = load_records_from_manifest(manifest_path)
    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    log.info("Loaded %d records from %s", len(records), manifest_path)

    dataset = ImageFolderDataset(records, training=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = VQGANTokenizer()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    total_batches = len(loader) if hasattr(loader, "__len__") else 0
    progress = Progress(total=max(0, args.epochs * total_batches), log_every=max(1, int(args.log_every)), label="vq")
    progress.start()

    ema_loss: float | None = None
    step = 0
    for epoch in range(1, args.epochs + 1):
        log.info("Epoch %d/%d starting", epoch, args.epochs)
        for imgs, _ in loader:
            step += 1
            imgs = imgs.permute(0, 3, 1, 2)
            recon, vq = model(imgs)
            loss = loss_fn(recon, imgs) + vq.loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_value = float(loss.detach().cpu().item())
            ema_loss = loss_value if ema_loss is None else (0.95 * ema_loss + 0.05 * loss_value)

            progress.update(1)
            if progress.should_log():
                msg, elapsed, eta = progress.stats()
                log.info("%s | loss %.4f (ema %.4f) | elapsed %s", msg, loss_value, ema_loss, format_duration(elapsed))
        log.info("Epoch %d/%d done", epoch, args.epochs)


if __name__ == "__main__":
    main()
