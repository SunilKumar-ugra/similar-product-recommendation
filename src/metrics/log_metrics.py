import json
from datetime import datetime
from pathlib import Path

METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)

def log_metrics(metrics: dict):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        **metrics
    }

    with open(METRICS_DIR / "metrics.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
