# import numpy as np
# from src.indexing.search import search

# embeddings = np.load("artifacts/embeddings.npy")
# product_ids = np.load("artifacts/product_ids.npy")

# query_idx = 100
# query_emb = embeddings[query_idx]

# idxs, scores = search(query_emb, k=5)

# print("Query product:", product_ids[query_idx])
# print("Similar products:", product_ids[idxs])
# print("Scores:", scores)

# import numpy as np

# emb = np.load("artifacts/embeddings.npy")
# ids = np.load("artifacts/product_ids.npy")

# print(emb.shape)           # (44419, 2048)
# print(ids.shape)           # (44419,)
# print(np.isnan(emb).any()) # False
# print(np.linalg.norm(emb[0]))  # ~1.0


# import numpy as np

# emb = np.load("artifacts/efficientnet/embeddings.npy")
# ids = np.load("artifacts/efficientnet/product_ids.npy")

# print(emb.shape)              # (44419, 1280)
# print(ids.shape)              # (44419,)
# print(np.isnan(emb).any())    # False
# print(np.linalg.norm(emb[0])) # ~1.0

# import numpy as np

# emb = np.load("artifacts/clip/embeddings.npy")
# ids = np.load("artifacts/clip/product_ids.npy")

# print(emb.shape)              # (44419, 512)
# print(ids.shape)              # (44419,)
# print(np.isnan(emb).any())    # False
# print(np.linalg.norm(emb[0])) # ~1.0

# from src.evaluation.evaluate import evaluate

# print("ResNet:", evaluate("artifacts/resnet", "data/processed/metadata.csv"))
# print("EfficientNet:", evaluate("artifacts/efficientnet", "data/processed/metadata.csv"))
# print("CLIP:", evaluate("artifacts/clip", "data/processed/metadata.csv"))



import numpy as np

emb = np.load("artifacts/hybrid/embeddings.npy")
ids = np.load("artifacts/hybrid/product_ids.npy")

print(emb.shape)              # (44419, 512)
print(ids.shape)              # (44419,)
print(np.isnan(emb).any())    # False
print(np.linalg.norm(emb[0])) # ~1.0
