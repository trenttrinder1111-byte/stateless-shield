import os
import json
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

    def _validate_path(self, filepath: str) -> str:
        """Security sandbox check to prevent path traversal exploits."""
        abs_path = os.path.abspath(filepath)
        base_dir = os.path.abspath(os.getcwd())
        if not abs_path.startswith(base_dir):
            raise ValueError("Security Error: Path traversal attempt blocked.")
        return abs_path

    def save(self, filepath_prefix: str):
        """Atomically and securely saves the index and payloads to disk."""
        safe_prefix = self._validate_path(filepath_prefix)
        index_path = f"{safe_prefix}.index"
        meta_path = f"{safe_prefix}.json"

        # 1. Atomic write for USearch binary index
        temp_index = f"{index_path}.tmp"
        self.index.save(temp_index)
        os.replace(temp_index, index_path)

        # 2. Atomic write for payload metadata
        temp_meta = f"{meta_path}.tmp"
        serializable_contents = {str(k): v for k, v in self.contents.items()}
        with open(temp_meta, 'w', encoding='utf-8') as f:
            json.dump(serializable_contents, f, ensure_ascii=False, indent=2)
        os.replace(temp_meta, meta_path)

    def load(self, filepath_prefix: str):
        """Securely loads the index and payloads from disk."""
        safe_prefix = self._validate_path(filepath_prefix)
        index_path = f"{safe_prefix}.index"
        meta_path = f"{safe_prefix}.json"

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Storage files missing or corrupted.")

        self.index.load(index_path)
        with open(meta_path, 'r', encoding='utf-8') as f:
            raw_contents = json.load(f)
            self.contents = {int(k): v for k, v in raw_contents.items()}
