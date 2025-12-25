import numpy as np
import pandas as pd
import faiss
import time
from collections import Counter
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity



def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return index

def intra_list_diversity(embeddings):
    sims = []
    for i, j in combinations(range(len(embeddings)), 2):
        sim = cosine_similarity(
            embeddings[i:i+1],
            embeddings[j:j+1]
        )[0][0]
        sims.append(1 - sim)
    return sum(sims) / len(sims)


def evaluate_model(artifact_dir, metadata_path, k=5, sample_size=500):
    emb = np.load(f"{artifact_dir}/embeddings.npy")
    ids = np.load(f"{artifact_dir}/product_ids.npy").astype(str)
    
    rng = np.random.default_rng(42)
    query_indices = rng.choice(len(emb), sample_size, replace=False)


    df = pd.read_csv(metadata_path, dtype={"product_id": str})
    df = df.set_index("product_id").loc[ids]

    index = build_index(emb)

    query_indices = np.random.choice(len(emb), sample_size, replace=False)

    precision_scores = []
    consistency_scores = []
    latencies = []
    diversities = []
    category_spreads = []


    for idx in query_indices:
        q_emb = emb[idx:idx+1]
        q_id = ids[idx]
        q_cat = df.loc[q_id]["category"]

        start = time.time()
        scores, neighbors = index.search(q_emb.astype("float32"), k + 1)
        latencies.append((time.time() - start) * 1000)

        neighbor_ids = ids[neighbors[0][1:]]
        neighbor_cats = df.loc[neighbor_ids]["category"].tolist()
        neighbor_embs = emb[neighbors[0][1:]]
        
        diversities.append(intra_list_diversity(neighbor_embs))
        category_spreads.append(
            len(set(neighbor_cats)) / k
        )

        hits = sum(1 for c in neighbor_cats if c == q_cat)
        precision_scores.append(hits / k)

        majority_cat = Counter(neighbor_cats).most_common(1)[0][0]
        consistency_scores.append(majority_cat == q_cat)

    # return {
    #     "Precision@5": round(float(np.mean(precision_scores)), 4),
    #     "CategoryConsistency": round(float(np.mean(consistency_scores)), 4),
    #     "AvgQueryLatency(ms)": round(float(np.mean(latencies)), 2),
    # }
    return {
        "Precision@5": round(float(np.mean(precision_scores)), 4),
        "CategoryConsistency": round(float(np.mean(consistency_scores)), 4),
        "ILD@5": round(float(np.mean(diversities)), 4),
        "CategorySpread@5": round(float(np.mean(category_spreads)), 4),
        "AvgQueryLatency(ms)": round(float(np.mean(latencies)), 2),
    }



if __name__ == "__main__":
    METADATA = "data/processed/metadata.csv"

    models = {
        "ResNet": "artifacts/resnet",
        "EfficientNet": "artifacts/efficientnet",
        "CLIP": "artifacts/clip_image",
        "Hybrid": "artifacts/hybrid",
    }

    for name, path in models.items():
        print(f"\n{name}")
        print(evaluate_model(path, METADATA))
