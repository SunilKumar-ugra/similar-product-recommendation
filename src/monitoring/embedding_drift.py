import numpy as np

def embedding_drift_score(ref_embs: np.ndarray, new_embs: np.ndarray) -> float:
    """
    Cosine distance between mean embedding vectors.
    """
    ref_mean = ref_embs.mean(axis=0)
    new_mean = new_embs.mean(axis=0)

    ref_mean /= np.linalg.norm(ref_mean)
    new_mean /= np.linalg.norm(new_mean)

    drift = 1 - np.dot(ref_mean, new_mean)
    return float(drift)
