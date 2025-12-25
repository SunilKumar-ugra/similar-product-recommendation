# src/ingestion/normalize.py
from PIL import Image
import os

def normalize_image(src_path, dst_path, size):
    img = Image.open(src_path).convert("RGB")
    img = img.resize(size)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    img.save(dst_path, "JPEG", quality=95)
''