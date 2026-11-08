import numpy as np
from usearch.index import Index

class StatelessShield:
    def __init__(self, vector_dim=128):
        self.vector_dim = vector_dim
        self.index = Index(ndim=vector_dim, metric="cos", dtype="f16")
        self.contents = {}

    def add(self, node_id: int, vector: np.ndarray, payload: str):
        vector = np.array(vector, dtype=np.float32)
        self.index.add(np.array([node_id], dtype=np.int64), vector.reshape(1, -1))
        self.contents[node_id] = payload

    def search(self, query_vector: np.ndarray, top_k=3):
        query_vector = np.array(query_vector, dtype=np.float32)
        matches = self.index.search(query_vector, top_k)
        return [(match.key, self.contents.get(match.key, "Unknown"), match.distance) for match in matches]
