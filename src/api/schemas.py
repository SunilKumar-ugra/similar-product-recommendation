from pydantic import BaseModel
from typing import List

class Recommendation(BaseModel):
    product_id: str
    score: float
    ann_score: float
    category_similarity: float
    category_match: bool



class RecommendationResponse(BaseModel):
    query_product_id: str
    model: str
    top_k: int
    results: List[Recommendation]
