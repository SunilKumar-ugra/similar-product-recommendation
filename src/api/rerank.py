# def rerank(
#     candidate_ids,
#     ann_scores,
#     metadata,
#     query_category,
#     w_sim=0.7,
#     w_cat=0.3
# ):
#     reranked = []

#     for pid, score in zip(candidate_ids, ann_scores):
#         cat = metadata.loc[pid]["category"]
#         cat_bonus = 1.0 if cat == query_category else 0.0

#         final_score = w_sim * score + w_cat * cat_bonus

#         reranked.append({
#             "product_id": pid,
#             "score": round(final_score, 4),
#             "ann_score": round(float(score), 4),
#             "category_match": bool(cat_bonus)
#         })

#     reranked.sort(key=lambda x: x["score"], reverse=True)
#     return reranked

# def rerank(candidate_ids, ann_scores, metadata, query_category, w_sim=0.7, w_cat=0.3):
#     reranked = []

#     for pid, score in zip(candidate_ids, ann_scores):
#         if pid not in metadata.index:
#             continue

#         cat = metadata.loc[pid]["category"]
#         cat_bonus = 1.0 if cat == query_category else 0.0

#         final_score = w_sim * score + w_cat * cat_bonus

#         reranked.append({
#             "product_id": pid,
#             "score": round(float(final_score), 4),
#             "ann_score": round(float(score), 4),
#             "category_match": bool(cat_bonus)
#         })

#     reranked.sort(key=lambda x: x["score"], reverse=True)
#     return reranked


from src.api.category_similarity import category_similarity

def rerank(
    candidate_ids,
    ann_scores,
    metadata,
    query_category,
    w_sim=0.6,
    w_cat=0.4
):
    reranked = []

    for pid, score in zip(candidate_ids, ann_scores):
        if pid not in metadata.index:
            continue

        cand_cat = metadata.loc[pid]["category"]

        cat_sim = category_similarity(query_category, cand_cat)
        cat_match = cand_cat == query_category

        final_score = w_sim * score + w_cat * cat_sim

        reranked.append({
            "product_id": pid,
            "score": float(final_score),
            "ann_score": float(score),
            "category_similarity": float(cat_sim),
            "category_match": bool(cat_match)
        })


    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked



