# src/modeling/clip_text/embed.py
import clip
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, _ = clip.load("ViT-B/32", device=DEVICE)

def encode_text(text: str):
    text = text.lower().strip()
    tokens = clip.tokenize([text]).to(DEVICE)

    with torch.no_grad():
        emb = MODEL.encode_text(tokens)
        emb = emb / emb.norm(dim=1, keepdim=True)

    return emb.cpu()
