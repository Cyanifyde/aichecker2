import io
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass
class LetterboxConfig:
    size: int = 512
    pad_mode: str = "reflect"


def _pad_image(img: Image.Image, size: int, pad_mode: str) -> Image.Image:
    width, height = img.size
    pad_w = size - width
    pad_h = size - height
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    if pad_mode == "reflect":
        arr = np.array(img)
        arr = np.pad(
            arr,
            ((top, bottom), (left, right), (0, 0)),
            mode="reflect",
        )
        return Image.fromarray(arr)
    if pad_mode == "edge":
        return ImageOps.expand(img, border=(left, top, right, bottom), fill=None)
    return ImageOps.expand(img, border=(left, top, right, bottom), fill=(0, 0, 0))


def letterbox_512(
    img: Image.Image,
    config: LetterboxConfig | None = None,
    interpolation: Image.Resampling | None = None,
) -> Image.Image:
    if config is None:
        config = LetterboxConfig()
    size = config.size
    if interpolation is None:
        interpolation = random.choice(
            [
                Image.Resampling.BILINEAR,
                Image.Resampling.BICUBIC,
                Image.Resampling.LANCZOS,
            ]
        )
    width, height = img.size
    scale = size / max(width, height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = img.resize((new_w, new_h), interpolation)
    return _pad_image(resized, size, config.pad_mode)


def random_augment(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))
    if random.random() < 0.5:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    if random.random() < 0.5:
        img = ImageOps.autocontrast(img)
    if random.random() < 0.3:
        img = ImageOps.equalize(img)
    return img


def jpeg_recompress(img: Image.Image, quality_range: Tuple[int, int] = (45, 95)) -> Image.Image:
    quality = random.randint(*quality_range)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")
