from typing import Dict, List

from pydantic import BaseModel


class TokenModel(BaseModel):
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


class InferenceResponse(BaseModel):
    prob_ai: float
    decision: str
    confidence: float
    ood_score: float
    tau_ai: float
    model_version: str
    tokens: List[TokenModel]
    image_hash: str


class FeedbackRequest(BaseModel):
    image_hash: str
    label: str


class FeedbackRecord(BaseModel):
    image_hash: str
    label: str
    updated_at: str


class HistoryRecord(BaseModel):
    image_hash: str
    prob_ai: float
    decision: str
    confidence: float
    ood_score: float
    created_at: str
