# src/indexing/build_index.py
import faiss
import numpy as np
import os

ARTIFACT_DIR = "artifacts"
INDEX_PATH = f"{ARTIFACT_DIR}/faiss.index"

def run():
    embeddings = np.load(f"{ARTIFACT_DIR}/embeddings.npy").astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    print(f"FAISS index built with {index.ntotal} vectors")

if __name__ == "__main__":
    run()
