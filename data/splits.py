import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def average_hash(img: Image.Image, hash_size: int = 8) -> str:
    img = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img)
    avg = pixels.mean()
    bits = pixels > avg
    bit_string = "".join("1" if bit else "0" for bit in bits.flatten())
    return hashlib.sha1(bit_string.encode()).hexdigest()


def group_by_phash(paths: List[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for path in paths:
        img = Image.open(path).convert("RGB")
        key = average_hash(img)
        groups.setdefault(key, []).append(path)
    return groups


def split_groups(groups: Dict[str, List[Path]], test_ratio: float = 0.1, seed: int = 7):
    keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    test_count = int(len(keys) * test_ratio)
    test_keys = set(keys[:test_count])
    train_keys = keys[test_count:]
    train_paths = [p for k in train_keys for p in groups[k]]
    test_paths = [p for k in test_keys for p in groups[k]]
    return train_paths, test_paths


def kfold_groups(groups: Dict[str, List[Path]], k: int = 5, seed: int = 7):
    keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    folds = np.array_split(keys, k)
    for i in range(k):
        val_keys = set(folds[i])
        train_keys = [k for j, fold in enumerate(folds) if j != i for k in fold]
        train_paths = [p for key in train_keys for p in groups[key]]
        val_paths = [p for key in val_keys for p in groups[key]]
        yield train_paths, val_paths
