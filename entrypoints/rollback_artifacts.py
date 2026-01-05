import shutil
from pathlib import Path

CURRENT = Path("artifacts")
BACKUP = Path("artifacts_backup")

if not BACKUP.exists():
    raise RuntimeError("❌ No backup artifacts available for rollback")

if CURRENT.exists():
    shutil.rmtree(CURRENT)

shutil.copytree(BACKUP, CURRENT)
print("⏪ Rollback completed")
