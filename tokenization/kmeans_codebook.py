from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass
class Codebook:
    centroids: np.ndarray

    def quantize(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dists = np.linalg.norm(embeddings[:, None, :] - self.centroids[None, :, :], axis=-1)
        ids = np.argmin(dists, axis=1)
        return ids, self.centroids[ids]


def build_codebook(embeddings: np.ndarray, num_clusters: int = 2048) -> Codebook:
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=2048, random_state=7)
    kmeans.fit(embeddings)
    return Codebook(centroids=kmeans.cluster_centers_)
