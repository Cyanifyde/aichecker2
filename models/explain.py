from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class TokenExplanation:
    discrete_id: int
    x: float
    y: float
    scale: float
    proposal_confidence: float
    importance: float
    proposal_type: str
    proposal_score_raw: float
    saliency_components: Dict[str, float]
    neighborhood_stats: Dict[str, float]
    model_contrib: Dict[str, float]
    reason: str


def summarize_reason(token: TokenExplanation) -> str:
    top_component = max(token.saliency_components, key=token.saliency_components.get, default="")
    attn = token.model_contrib.get("attn_score", 0.0)
    grad = token.model_contrib.get("grad_score", 0.0)
    return f"Strong {top_component} with attention {attn:.2f} and gradient {grad:.2f}."


def compute_importance(attn_scores: np.ndarray, grad_scores: np.ndarray) -> np.ndarray:
    attn_norm = (attn_scores - attn_scores.min()) / (attn_scores.ptp() + 1e-6)
    grad_norm = (grad_scores - grad_scores.min()) / (grad_scores.ptp() + 1e-6)
    return 0.6 * attn_norm + 0.4 * grad_norm


def build_explanations(
    ids: np.ndarray,
    coords: np.ndarray,
    proposal_conf: np.ndarray,
    proposal_type: List[str],
    proposal_score_raw: np.ndarray,
    saliency_components: List[Dict[str, float]],
    neighborhood_stats: List[Dict[str, float]],
    attn_scores: np.ndarray,
    grad_scores: np.ndarray,
) -> List[TokenExplanation]:
    importance = compute_importance(attn_scores, grad_scores)
    output: List[TokenExplanation] = []
    for idx, token_id in enumerate(ids):
        model_contrib = {
            "attn_score": float(attn_scores[idx]),
            "grad_score": float(grad_scores[idx]),
        }
        model_contrib["final_importance"] = float(importance[idx])
        token = TokenExplanation(
            discrete_id=int(token_id),
            x=float(coords[idx, 0]),
            y=float(coords[idx, 1]),
            scale=float(coords[idx, 2]),
            proposal_confidence=float(proposal_conf[idx]),
            importance=float(importance[idx]),
            proposal_type=proposal_type[idx],
            proposal_score_raw=float(proposal_score_raw[idx]),
            saliency_components=saliency_components[idx],
            neighborhood_stats=neighborhood_stats[idx],
            model_contrib=model_contrib,
            reason="",
        )
        token.reason = summarize_reason(token)
        output.append(token)
    return output
