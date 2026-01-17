from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class SlotToken:
    x: float
    y: float
    scale: float
    score: float


class SlotAttention(nn.Module):
    def __init__(self, num_slots: int = 6, dim: int = 64, iters: int = 3):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.scale = dim**-0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, d = x.shape
        slots = self.slots_mu + torch.exp(self.slots_logsigma) * torch.randn(
            b, self.num_slots, d, device=x.device
        )
        k = self.to_k(x)
        v = self.to_v(x)
        for _ in range(self.iters):
            q = self.to_q(slots)
            attn_logits = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = attn_logits.softmax(dim=-1)
            updates = torch.einsum("bji,bjd->bid", attn, v)
            slots = self.gru(
                updates.reshape(-1, d), slots.reshape(-1, d)
            ).reshape(b, -1, d)
            slots = slots + self.mlp(slots)
        return slots, attn


def slots_from_features(feature_map: np.ndarray, num_slots: int = 6) -> List[SlotToken]:
    h, w, c = feature_map.shape
    features = torch.from_numpy(feature_map.reshape(-1, c)).unsqueeze(0).float()
    slot_attention = SlotAttention(num_slots=num_slots, dim=c)
    with torch.no_grad():
        _, attn = slot_attention(features)
    attn = attn.squeeze(0).numpy()
    tokens = []
    for slot_idx in range(num_slots):
        weights = attn[slot_idx].reshape(h, w)
        total = weights.sum() + 1e-6
        ys, xs = np.mgrid[0:h, 0:w]
        cx = float((weights * xs).sum() / total) / w
        cy = float((weights * ys).sum() / total) / h
        var = float(((weights * ((xs - cx * w) ** 2 + (ys - cy * h) ** 2)).sum() / total))
        scale = np.sqrt(var) / max(h, w)
        tokens.append(SlotToken(x=cx, y=cy, scale=float(scale), score=float(weights.max())))
    return tokens
