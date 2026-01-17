from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class VQOutput:
    quantized: torch.Tensor
    indices: torch.Tensor
    loss: torch.Tensor


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int = 2048, embedding_dim: int = 128, beta: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: torch.Tensor) -> VQOutput:
        flat = z.reshape(-1, self.embedding_dim)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(1)
        )
        indices = dist.argmin(1)
        quantized = self.embeddings(indices).reshape_as(z)
        loss = ((quantized.detach() - z) ** 2).mean() + self.beta * ((quantized - z.detach()) ** 2).mean()
        quantized = z + (quantized - z).detach()
        return VQOutput(quantized=quantized, indices=indices.view(z.shape[0], -1), loss=loss)


class VQGANTokenizer(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_embeddings: int = 2048):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embedding_dim, 3, padding=1),
        )
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, VQOutput]:
        z = self.encoder(x)
        vq = self.quantizer(z)
        recon = self.decoder(vq.quantized)
        return recon, vq
