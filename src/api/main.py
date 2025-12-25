from fastapi import FastAPI, Query, HTTPException
import numpy as np
import pandas as pd

from src.api.faiss_index import FaissIndex
from src.api.rerank import rerank
from src.api.schemas import RecommendationResponse
from src.api.mmr import mmr_rerank


app = FastAPI(title="Similarity Search API")

# ----------------------------
# Load resources ONCE
# ----------------------------
METADATA = pd.read_csv("data/processed/metadata.csv", dtype={"product_id": str})
METADATA = METADATA.set_index("product_id")

MODELS = {
    "clip": FaissIndex(
        "artifacts/clip_image/embeddings.npy",
        "artifacts/clip_image/product_ids.npy"
    ),
    "hybrid": FaissIndex(
        "artifacts/hybrid/embeddings.npy",
        "artifacts/hybrid/product_ids.npy"
    )
}

# ----------------------------
# API
# ----------------------------

@app.get("/recommend", response_model=RecommendationResponse)
def recommend(
    product_id: str,
    model: str = Query("clip", enum=["clip", "hybrid"]),
    top_k: int = 5,
    lambda_diversity: float = 0.7
):
    print("REQUEST:", product_id, model)

    index = MODELS[model]

    # ----------------------------
    # Validate product
    # ----------------------------
    if product_id not in METADATA.index:
        raise HTTPException(status_code=400, detail="Invalid product_id")

    if product_id not in index.id_to_idx:
        raise HTTPException(status_code=400, detail="Product not indexed")

    query_idx = index.id_to_idx[product_id]
    query_category = METADATA.loc[product_id]["category"]
    query_emb = index.embeddings[query_idx:query_idx + 1]

    # ----------------------------
    # ANN recall
    # ----------------------------
    candidate_ids, ann_scores = index.search(query_idx, top_k=50)

    # ----------------------------
    # Build candidate embeddings safely
    # ----------------------------
    candidate_embs = np.array([
        index.embeddings[index.id_to_idx[pid]]
        for pid in candidate_ids
        if pid in index.id_to_idx
    ])

    # Align scores with IDs
    score_map = dict(zip(candidate_ids, ann_scores))

    # ----------------------------
    # MMR re-ranking
    # ----------------------------
    mmr_ids = mmr_rerank(
        query_emb=query_emb,
        candidate_embs=candidate_embs,
        candidate_ids=candidate_ids,
        candidate_scores=ann_scores,
        lambda_param=lambda_diversity,
        top_k=top_k
    )

    # ----------------------------
    # Final semantic + category rerank
    # ----------------------------
    reranked = rerank(
        mmr_ids,
        [score_map[pid] for pid in mmr_ids],
        METADATA,
        query_category
    )

    print("Sample result:", reranked[0])

    return {
        "query_product_id": product_id,
        "model": model,
        "top_k": top_k,
        "results": reranked[:top_k]
    }


