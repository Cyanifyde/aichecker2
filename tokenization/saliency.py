from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
from scipy import fftpack


@dataclass
class SaliencyToken:
    x: float
    y: float
    scale: float
    score: float
    components: Dict[str, float]


def _entropy_map(gray: np.ndarray, ksize: int = 9) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    hist = cv2.calcHist([blurred], [0], None, [16], [0, 256])
    prob = hist / np.sum(hist)
    entropy = -np.sum(prob * np.log2(prob + 1e-9))
    return np.full_like(gray, entropy)


def _fft_energy(gray: np.ndarray) -> np.ndarray:
    f = fftpack.fft2(gray)
    fshift = fftpack.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    return magnitude / magnitude.max()


def saliency_tokens(img: np.ndarray, max_points: int = 256) -> List[SaliencyToken]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    entropy = _entropy_map(gray)
    fft_energy = _fft_energy(gray)

    score_map = (
        cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)
        + cv2.normalize(np.abs(lap), None, 0, 1, cv2.NORM_MINMAX)
        + cv2.normalize(entropy, None, 0, 1, cv2.NORM_MINMAX)
        + fft_energy
    )
    score_map = score_map / score_map.max()
    flat = score_map.flatten()
    idxs = np.argpartition(flat, -max_points)[-max_points:]
    h, w = gray.shape
    tokens = []
    for idx in idxs:
        y, x = divmod(int(idx), w)
        score = float(score_map[y, x])
        tokens.append(
            SaliencyToken(
                x=x / w,
                y=y / h,
                scale=3.0 / max(w, h),
                score=score,
                components={
                    "grad_energy": float(grad_mag[y, x]),
                    "entropy": float(entropy[y, x]),
                    "laplacian": float(lap[y, x]),
                    "fft_energy": float(fft_energy[y, x]),
                },
            )
        )
    return tokens
