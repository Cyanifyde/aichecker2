from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TokenProposal:
    proposal_type: str
    x: float
    y: float
    scale: float
    score: float
    saliency_components: dict | None = None


def _nms(proposals: List[TokenProposal], radius: float = 0.03) -> List[TokenProposal]:
    if not proposals:
        return []
    proposals = sorted(proposals, key=lambda p: p.score, reverse=True)
    kept: List[TokenProposal] = []
    for prop in proposals:
        if all((prop.x - k.x) ** 2 + (prop.y - k.y) ** 2 > radius**2 for k in kept):
            kept.append(prop)
    return kept


def fuse_proposals(
    proposals: List[TokenProposal],
    n_min: int = 128,
    n_max: int = 512,
) -> List[TokenProposal]:
    deduped = _nms(proposals)
    if len(deduped) > n_max:
        deduped = deduped[:n_max]
    if len(deduped) < n_min:
        extra = np.random.choice(deduped, size=n_min - len(deduped), replace=True)
        deduped.extend(extra.tolist())
    return deduped
