import numpy as np
import pandas as pd
import faiss
from collections import Counter
import time

def evaluate(artifact_dir, metadata_path, k=5, sample_size=500):
    # Load data
    embeddings = np.load(f"{artifact_dir}/embeddings.npy").astype("float32")
    product_ids = np.load(f"{artifact_dir}/product_ids.npy")
    df = pd.read_csv(metadata_path, dtype={"product_id": str}).set_index("product_id")

    #df = pd.read_csv(metadata_path).set_index("product_id")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Sample queries
    indices = np.random.choice(len(embeddings), sample_size, replace=False)

    precision_hits = []
    category_consistency = []
    latencies = []

    for idx in indices:
        query_emb = embeddings[idx:idx+1]
        query_pid = product_ids[idx]
        query_cat = df.loc[str(query_pid)]["category"]

        start = time.time()
        scores, neighbors = index.search(query_emb, k+1)
        latencies.append(time.time() - start)

        neighbor_ids = product_ids[neighbors[0][1:]]  # exclude self
        neighbor_cats = df.loc[neighbor_ids.astype(str)]["category"].tolist()

        hits = sum(1 for c in neighbor_cats if c == query_cat)
        precision_hits.append(hits / k)

        most_common_cat = Counter(neighbor_cats).most_common(1)[0][0]
        category_consistency.append(most_common_cat == query_cat)

    return {
        "Precision@{}".format(k): round(np.mean(precision_hits), 4),
        "CategoryConsistency": round(np.mean(category_consistency), 4),
        "AvgQueryLatency(ms)": round(np.mean(latencies) * 1000, 2)
    }


evaluate("artifacts/resnet", "data/processed/metadata.csv")
#evaluate("artifacts/efficientnet", "data/processed/metadata.csv")
