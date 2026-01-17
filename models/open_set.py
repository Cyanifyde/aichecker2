from dataclasses import dataclass

import numpy as np


@dataclass
class Decision:
    prob_ai: float
    confidence: float
    ood_score: float
    decision: str


def decide(
    prob_ai: float,
    ood_score: float,
    tau_ai: float,
    tau_ood: float,
    tau_unknown: float,
) -> Decision:
    confidence = max(prob_ai, 1 - prob_ai)
    if ood_score >= tau_ood or confidence < tau_unknown:
        return Decision(prob_ai=prob_ai, confidence=confidence, ood_score=ood_score, decision="UNKNOWN")
    if prob_ai >= tau_ai:
        return Decision(prob_ai=prob_ai, confidence=confidence, ood_score=ood_score, decision="AI")
    return Decision(prob_ai=prob_ai, confidence=confidence, ood_score=ood_score, decision="NON_AI")


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / max(temperature, 1e-6)
