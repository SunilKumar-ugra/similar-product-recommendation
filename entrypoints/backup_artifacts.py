import shutil
from pathlib import Path

CURRENT = Path("artifacts")
BACKUP = Path("artifacts_backup")

if BACKUP.exists():
    shutil.rmtree(BACKUP)

if CURRENT.exists():
    shutil.copytree(CURRENT, BACKUP)
    print("✅ Artifacts backed up")
else:
    print("⚠️ No artifacts to back up")
