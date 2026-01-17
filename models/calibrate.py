import numpy as np


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    temperature = 1.0
    best_loss = float("inf")
    for temp in np.linspace(0.5, 5.0, 20):
        scaled = logits / temp
        probs = 1 / (1 + np.exp(-scaled))
        loss = -(labels * np.log(probs + 1e-9) + (1 - labels) * np.log(1 - probs + 1e-9)).mean()
        if loss < best_loss:
            best_loss = loss
            temperature = temp
    return float(temperature)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / max(temperature, 1e-6)
