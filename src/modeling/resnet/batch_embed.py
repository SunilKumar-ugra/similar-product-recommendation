# src/modeling/batch_embed.py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os

from model import ResNetEmbedding
from embed import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

METADATA_PATH = "data/processed/metadata.csv"
ARTIFACT_DIR = "artifacts"

def run():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = pd.read_csv(METADATA_PATH)

    model = ResNetEmbedding().to(DEVICE)
    model.eval()

    embeddings = []
    product_ids = []

    batch_images = []
    batch_ids = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img = load_image(row["image_path"])
            batch_images.append(img)
            batch_ids.append(row["product_id"])

            if len(batch_images) == BATCH_SIZE:
                batch_tensor = torch.stack(batch_images).to(DEVICE)
                batch_emb = model(batch_tensor).cpu().numpy()

                embeddings.append(batch_emb)
                product_ids.extend(batch_ids)

                batch_images, batch_ids = [], []

        # flush last batch
        if batch_images:
            batch_tensor = torch.stack(batch_images).to(DEVICE)
            batch_emb = model(batch_tensor).cpu().numpy()
            embeddings.append(batch_emb)
            product_ids.extend(batch_ids)

    embeddings = np.vstack(embeddings).astype("float32")
    product_ids = np.array(product_ids)

    np.save(f"{ARTIFACT_DIR}/embeddings.npy", embeddings)
    np.save(f"{ARTIFACT_DIR}/product_ids.npy", product_ids)

    with open(f"{ARTIFACT_DIR}/embedding_meta.json", "w") as f:
        json.dump({
            "model": "resnet50",
            "embedding_dim": embeddings.shape[1],
            "normalized": True,
            "num_products": len(product_ids)
        }, f, indent=2)

    print("Embeddings generated:", embeddings.shape)

if __name__ == "__main__":
    run()
