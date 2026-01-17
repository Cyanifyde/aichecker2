import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import LetterboxConfig, letterbox_512, random_augment


@dataclass
class ImageRecord:
    path: Path
    label: int


class ImageFolderDataset(Dataset):
    def __init__(self, records: Iterable[ImageRecord], training: bool = True):
        self.records = list(records)
        self.training = training
        self.letterbox_config = LetterboxConfig()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        record = self.records[idx]
        img = Image.open(record.path).convert("RGB")
        img = letterbox_512(img, self.letterbox_config)
        if self.training:
            img = random_augment(img)
        arr = np.asarray(img).astype(np.float32) / 255.0
        return arr, record.label


def load_records_from_manifest(manifest_path: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        base = manifest_path.parent
        for row in reader:
            path = base / row["path"]
            label = int(row["label"])
            records.append(ImageRecord(path=path, label=label))
    return records
