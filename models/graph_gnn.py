import torch
from torch import nn


class GraphClassifier(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.head = nn.Linear(dim, 1)
        self.ood_head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, _ = self.attn(x, x, x)
        pooled = attn_out.mean(dim=1)
        pooled = pooled + self.mlp(pooled)
        return self.head(pooled).squeeze(-1), self.ood_head(pooled).squeeze(-1)
