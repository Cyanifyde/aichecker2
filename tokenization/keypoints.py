from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class KeypointToken:
    x: float
    y: float
    scale: float
    response: float


def detect_keypoints(img: np.ndarray, max_points: int = 256) -> List[KeypointToken]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fast = cv2.FastFeatureDetector_create(threshold=18, nonmaxSuppression=True)
    keypoints = fast.detect(gray, None)
    keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)[:max_points]
    tokens = []
    h, w = gray.shape
    for kp in keypoints:
        tokens.append(
            KeypointToken(
                x=kp.pt[0] / w,
                y=kp.pt[1] / h,
                scale=max(1.0, kp.size / max(w, h)),
                response=kp.response,
            )
        )
    return tokens
