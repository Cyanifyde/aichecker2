from typing import List

import cv2
import numpy as np
from PIL import Image


def canny_trace(img: Image.Image) -> Image.Image:
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(arr, 80, 160)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def xdog_lineart(img: Image.Image) -> Image.Image:
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(arr, (0, 0), 0.8)
    g2 = cv2.GaussianBlur(arr, (0, 0), 3.0)
    xdog = g1 - 0.95 * g2
    xdog = (xdog > 0.02).astype(np.uint8) * 255
    return Image.fromarray(cv2.cvtColor(xdog, cv2.COLOR_GRAY2RGB))


def traced_variants(img: Image.Image) -> List[Image.Image]:
    return [canny_trace(img), xdog_lineart(img)]
