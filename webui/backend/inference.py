import hashlib
import io
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from PIL import Image

from models.explain import build_explanations
from models.open_set import decide
from tokenization.keypoints import detect_keypoints
from tokenization.patches import batch_extract
from tokenization.preprocess_512 import preprocess_image
from tokenization.saliency import saliency_tokens
from tokenization.slot_attention import slots_from_features
from tokenization.token_fusion import TokenProposal, fuse_proposals


@dataclass
class InferenceResult:
    image_bytes: bytes
    image_hash: str
    prob_ai: float
    decision: str
    confidence: float
    ood_score: float
    tokens: list


def _score_image(img: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    variance = float(lap.var())
    edge_density = float((np.abs(lap) > 10).mean())
    prob_ai = 1 / (1 + np.exp(-(variance - 5) / 5))
    ood_score = min(1.0, edge_density * 2.0)
    return prob_ai, ood_score


def _discretize(patches: np.ndarray, codebook_size: int = 2048) -> np.ndarray:
    means = patches.mean(axis=(1, 2, 3))
    ids = (means * codebook_size).astype(int) % codebook_size
    return ids


def run_inference(img: Image.Image, tau_ai: float, tau_ood: float, tau_unknown: float) -> InferenceResult:
    normalized = preprocess_image(img)
    buffer = io.BytesIO()
    normalized.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    arr = np.asarray(normalized)
    keypoints = detect_keypoints(arr)
    saliency = saliency_tokens(arr)
    downsample = cv2.resize(arr, (16, 16), interpolation=cv2.INTER_AREA)
    slots = slots_from_features(downsample)

    proposals: List[TokenProposal] = []
    for kp in keypoints:
        proposals.append(
            TokenProposal(
                proposal_type="keypoint",
                x=kp.x,
                y=kp.y,
                scale=kp.scale,
                score=kp.response,
            )
        )
    for token in saliency:
        proposals.append(
            TokenProposal(
                proposal_type="saliency",
                x=token.x,
                y=token.y,
                scale=token.scale,
                score=token.score,
                saliency_components=token.components,
            )
        )
    for slot in slots:
        proposals.append(
            TokenProposal(
                proposal_type="slot",
                x=slot.x,
                y=slot.y,
                scale=slot.scale,
                score=slot.score,
            )
        )

    fused = fuse_proposals(proposals)
    coords = np.array([[p.x, p.y, p.scale] for p in fused])
    patches = batch_extract(normalized, [(p.x, p.y, p.scale) for p in fused])
    ids = _discretize(patches)
    attn_scores = np.array([p.score for p in fused])
    grad_scores = np.array([p.score * 0.8 for p in fused])
    proposal_conf = np.array([p.score for p in fused])
    proposal_score_raw = np.array([p.score for p in fused])
    proposal_type = [p.proposal_type for p in fused]
    saliency_components = [p.saliency_components or {} for p in fused]
    neighborhood_stats = [{"mean_neighbor_similarity": float(np.random.rand())} for _ in fused]

    prob_ai, ood_score = _score_image(arr)
    decision = decide(prob_ai, ood_score, tau_ai, tau_ood, tau_unknown)
    tokens = build_explanations(
        ids=ids,
        coords=coords,
        proposal_conf=proposal_conf,
        proposal_type=proposal_type,
        proposal_score_raw=proposal_score_raw,
        saliency_components=saliency_components,
        neighborhood_stats=neighborhood_stats,
        attn_scores=attn_scores,
        grad_scores=grad_scores,
    )

    return InferenceResult(
        image_bytes=image_bytes,
        image_hash=image_hash,
        prob_ai=decision.prob_ai,
        decision=decision.decision,
        confidence=decision.confidence,
        ood_score=decision.ood_score,
        tokens=tokens,
    )
