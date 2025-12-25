import numpy as np
import pandas as pd
import json
import os

ARTIFACT_DIR = "artifacts/hybrid"
IMG_DIR = "artifacts/clip_image"
TXT_DIR = "artifacts/clip_text"
METADATA_PATH = "data/processed/metadata.csv"

W_IMG = 0.6
W_TXT = 0.3
W_CAT = 0.1


def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def run():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Load embeddings
    img_emb = np.load(f"{IMG_DIR}/embeddings.npy")
    txt_emb = np.load(f"{TXT_DIR}/embeddings.npy")

    img_ids = np.load(f"{IMG_DIR}/product_ids.npy").astype(str)
    txt_ids = np.load(f"{TXT_DIR}/product_ids.npy").astype(str)

    # Hard alignment check
    assert (img_ids == txt_ids).all(), "Image/Text IDs misaligned"

    # Load metadata
    df = pd.read_csv(METADATA_PATH, dtype={"product_id": str})
    df = df.set_index("product_id").loc[img_ids]

    # Build category embeddings (simple, deterministic)
    categories = df["category"].astype("category")
    cat_codes = categories.cat.codes.values

    num_cats = categories.cat.categories.size
    cat_emb = np.eye(num_cats)[cat_codes]  # one-hot
    cat_emb = l2_normalize(cat_emb)

    # Project category to 512-D (cheap + stable)
    rng = np.random.default_rng(42)
    proj = rng.normal(0, 1, size=(cat_emb.shape[1], 512))
    cat_emb = cat_emb @ proj
    cat_emb = l2_normalize(cat_emb)

    # Hybrid fusion
    hybrid = (
        W_IMG * img_emb +
        W_TXT * txt_emb +
        W_CAT * cat_emb
    )

    hybrid = l2_normalize(hybrid)

    # Save artifacts
    np.save(f"{ARTIFACT_DIR}/embeddings.npy", hybrid.astype("float32"))
    np.save(f"{ARTIFACT_DIR}/product_ids.npy", img_ids)

    with open(f"{ARTIFACT_DIR}/embedding_meta.json", "w") as f:
        json.dump({
            "image_weight": W_IMG,
            "text_weight": W_TXT,
            "category_weight": W_CAT,
            "embedding_dim": 512,
            "num_products": len(img_ids)
        }, f, indent=2)

    print("✅ Hybrid embeddings generated:", hybrid.shape)


if __name__ == "__main__":
    run()
