from typing import List, Tuple

import numpy as np
from PIL import Image


def extract_patch(img: Image.Image, x: float, y: float, scale: float, size: int = 64) -> Image.Image:
    width, height = img.size
    cx = int(x * width)
    cy = int(y * height)
    radius = int(max(4, scale * max(width, height)))
    left = max(0, cx - radius)
    right = min(width, cx + radius)
    top = max(0, cy - radius)
    bottom = min(height, cy + radius)
    patch = img.crop((left, top, right, bottom))
    return patch.resize((size, size), Image.Resampling.BICUBIC)


def batch_extract(img: Image.Image, coords: List[Tuple[float, float, float]], size: int = 64) -> np.ndarray:
    patches = [extract_patch(img, x, y, scale, size=size) for x, y, scale in coords]
    arr = np.stack([np.asarray(p).astype(np.float32) / 255.0 for p in patches])
    return arr
