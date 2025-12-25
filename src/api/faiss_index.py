import faiss
import numpy as np

class FaissIndex:
    def __init__(self, embeddings_path, ids_path):
        # Load data
        self.embeddings = np.load(embeddings_path).astype("float32")
        self.ids = np.load(ids_path).astype(str)

        # Build ID → index map (CRITICAL)
        self.id_to_idx = {
            pid: i for i, pid in enumerate(self.ids)
        }

        # FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def search(self, query_idx, top_k=50):
        """
        Returns:
        - candidate_ids: list[str]
        - scores: list[float]
        """
        q_emb = self.embeddings[query_idx:query_idx + 1]
        scores, neighbors = self.index.search(q_emb, top_k + 1)

        # Convert FAISS indices → product IDs
        neighbor_ids = [
            self.ids[i] for i in neighbors[0][1:]
        ]

        return neighbor_ids, scores[0][1:]
