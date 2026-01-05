import subprocess

print("Building embeddings...")
subprocess.run(["python", "src/modeling/clip_image/batch_embed.py"], check=True)
subprocess.run(["python", "src/modeling/hybrid/batch_embed.py"], check=True)

print("Building FAISS indexes...")
subprocess.run(["python", "src/indexing/build_faiss_index.py"], check=True)

print("Artifacts built successfully.")
