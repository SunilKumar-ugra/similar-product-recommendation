import clip
import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, _ = clip.load("ViT-B/32", device=DEVICE)

_category_cache = {}

def get_category_embedding(category: str):
    if category not in _category_cache:
        tokens = clip.tokenize([category.lower()]).to(DEVICE)
        with torch.no_grad():
            emb = MODEL.encode_text(tokens)
            emb = emb / emb.norm(dim=1, keepdim=True)
        _category_cache[category] = emb.cpu().numpy()[0]
    return _category_cache[category]


def category_similarity(cat1: str, cat2: str) -> float:
    e1 = get_category_embedding(cat1)
    e2 = get_category_embedding(cat2)
    return float(np.dot(e1, e2))
