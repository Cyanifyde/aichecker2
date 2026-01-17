from typing import List

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def heavy_edit_proxies(img: Image.Image) -> List[Image.Image]:
    variants = []
    variants.append(ImageOps.posterize(img, bits=3))
    variants.append(ImageOps.solarize(img, threshold=96))
    variants.append(img.filter(ImageFilter.GaussianBlur(radius=3.0)))
    variants.append(img.filter(ImageFilter.DETAIL))
    return variants


def noise_proxy(img: Image.Image, sigma: float = 0.08) -> Image.Image:
    arr = np.asarray(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))
