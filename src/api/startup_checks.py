import os
import numpy as np
from pathlib import Path

def validate_artifacts(emb_path: str, ids_path: str, model_name: str):
    if not Path(emb_path).exists():
        raise RuntimeError(f"[{model_name}] Embeddings file missing: {emb_path}")

    if not Path(ids_path).exists():
        raise RuntimeError(f"[{model_name}] IDs file missing: {ids_path}")

    embeddings = np.load(emb_path)
    ids = np.load(ids_path)

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"[{model_name}] Embeddings must be 2D, got shape {embeddings.shape}"
        )

    if len(embeddings) != len(ids):
        raise RuntimeError(
            f"[{model_name}] Embeddings count ({len(embeddings)}) "
            f"does not match IDs count ({len(ids)})"
        )

    if embeddings.size == 0:
        raise RuntimeError(f"[{model_name}] Embeddings file is empty")

    print(f"✅ [{model_name}] Artifacts validated successfully")
