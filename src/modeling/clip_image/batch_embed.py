# src/modeling/clip/batch_embed.py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os

from embed import encode_image

ARTIFACT_DIR = "artifacts/clip_image"
METADATA_PATH = "data/processed/metadata.csv"

def run():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = pd.read_csv(METADATA_PATH)

    embeddings = []
    product_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        emb = encode_image(row["image_path"])
        embeddings.append(emb.numpy())
        product_ids.append(row["product_id"])

    embeddings = np.vstack(embeddings).astype("float32")
    product_ids = np.array(product_ids)

    np.save(f"{ARTIFACT_DIR}/embeddings.npy", embeddings)
    np.save(f"{ARTIFACT_DIR}/product_ids.npy", product_ids)

    with open(f"{ARTIFACT_DIR}/embedding_meta.json", "w") as f:
        json.dump({
            "model": "CLIP ViT-B/32 (image)",
            "embedding_dim": embeddings.shape[1],
            "normalized": True,
            "num_products": len(product_ids)
        }, f, indent=2)

    print("CLIP image embeddings generated:", embeddings.shape)

if __name__ == "__main__":
    run()
