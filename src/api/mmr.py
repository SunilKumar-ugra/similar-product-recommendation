import numpy as np

def mmr_rerank(
    query_emb,
    candidate_embs,
    candidate_ids,
    candidate_scores,
    lambda_param=0.7,
    top_k=5
):
    """
    MMR re-ranking.
    query_emb: (1, D)
    candidate_embs: (N, D)
    candidate_scores: ANN similarity scores
    """

    selected = []
    selected_indices = []

    candidates = list(range(len(candidate_ids)))

    for _ in range(top_k):
        mmr_scores = []

        for i in candidates:
            relevance = candidate_scores[i]

            if not selected_indices:
                diversity_penalty = 0.0
            else:
                sims = [
                    np.dot(candidate_embs[i], candidate_embs[j])
                    for j in selected_indices
                ]
                diversity_penalty = max(sims)

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * diversity_penalty
            )

            mmr_scores.append((mmr_score, i))

        _, best_idx = max(mmr_scores, key=lambda x: x[0])
        selected.append(candidate_ids[best_idx])
        selected_indices.append(best_idx)
        candidates.remove(best_idx)

    return selected
