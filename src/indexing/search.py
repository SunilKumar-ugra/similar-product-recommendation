# src/indexing/search.py
import faiss
import numpy as np

ARTIFACT_DIR = "artifacts"
INDEX_PATH = f"{ARTIFACT_DIR}/faiss.index"

index = faiss.read_index(INDEX_PATH)

def search(query_embedding, k=5):
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    scores, indices = index.search(query_embedding.astype("float32"), k)
    return indices[0], scores[0]
