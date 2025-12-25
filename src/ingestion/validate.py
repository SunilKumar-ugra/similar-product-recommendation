# src/ingestion/validate.py
from PIL import Image

def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
