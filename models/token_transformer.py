import torch
from torch import nn


class TokenTransformer(nn.Module):
    def __init__(self, vocab_size: int = 2048, dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_mlp = nn.Sequential(nn.Linear(4, dim), nn.ReLU(), nn.Linear(dim, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, ids: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        token = self.token_emb(ids)
        pos = self.pos_mlp(coords)
        x = token + pos
        return self.encoder(x)
