import yaml
import numpy as np
import pandas as pd

from embedding_drift import embedding_drift_score
from data_drift import population_stability_index
from performance_drift import precision_drop

print("🔍 Running drift checks...")

# Load thresholds
with open("src/monitoring/thresholds.yaml") as f:
    thresholds = yaml.safe_load(f)

# Load reference artifacts
ref_embs = np.load("artifacts/baseline/clip_embeddings.npy")
new_embs = np.load("artifacts/clip_image/embeddings.npy")

ref_meta = pd.read_csv("artifacts/baseline/metadata.csv")
new_meta = pd.read_csv("data/processed/metadata.csv")

# ---- Embedding Drift ----
emb_drift = embedding_drift_score(ref_embs, new_embs)
print(f"Embedding drift score: {emb_drift:.3f}")

# ---- Data Drift ----
psi = population_stability_index(
    ref_meta["category"],
    new_meta["category"]
)
print(f"Category PSI: {psi:.3f}")

# ---- Evaluate thresholds ----
if emb_drift > thresholds["embedding_drift"]["critical"]:
    raise RuntimeError("🚨 CRITICAL embedding drift detected")

if psi > thresholds["category_psi"]["critical"]:
    raise RuntimeError("🚨 CRITICAL category drift detected")

print("✅ Drift within acceptable limits")

import sys

# If everything is OK
print("DRIFT_STATUS=OK")
sys.exit(0)

from metrics.log_metrics import log_metrics

log_metrics({
    "embedding_drift": emb_drift,
    "category_psi": psi,
})
