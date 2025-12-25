# src/modeling/clip/embed.py
import clip
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)

def encode_image(image_path):
    image = PREPROCESS(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = MODEL.encode_image(image)
        emb = emb / emb.norm(dim=1, keepdim=True)
    return emb.cpu()
