# src/modeling/clip_text/batch_embed.py
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os

from embed import encode_text

ARTIFACT_DIR = "artifacts/clip_text"
METADATA_PATH = "data/processed/metadata.csv"

def run():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = pd.read_csv(METADATA_PATH, dtype={"product_id": str})

    embeddings = []
    product_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row["productDisplayName"]

        if not isinstance(text, str) or text.strip() == "":
            text = "unknown product"

        emb = encode_text(text)
        embeddings.append(emb.numpy())
        product_ids.append(row["product_id"])

    embeddings = np.vstack(embeddings).astype("float32")
    product_ids = np.array(product_ids)

    np.save(f"{ARTIFACT_DIR}/embeddings.npy", embeddings)
    np.save(f"{ARTIFACT_DIR}/product_ids.npy", product_ids)

    with open(f"{ARTIFACT_DIR}/embedding_meta.json", "w") as f:
        json.dump({
            "model": "CLIP ViT-B/32 (text)",
            "embedding_dim": embeddings.shape[1],
            "normalized": True,
            "num_products": len(product_ids)
        }, f, indent=2)

    print("CLIP text embeddings generated:", embeddings.shape)

if __name__ == "__main__":
    run()
