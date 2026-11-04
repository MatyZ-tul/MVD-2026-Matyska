import torch
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# 1. Detekce GPU pro PyTorch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Použité zařízení pro model: {device.upper()}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

print("1. Načítám Wikipedii...")
dataset = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")

print("2. Načítám model a přesouvám ho na GPU...")
# Klíčové: device=device zajistí, že model počítá na grafické kartě
model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", device=device)

print("3. Generuji vektory pro všechny články (běží na GPU)...")
texts = dataset["text"]

# Využijeme efektivní dávkování na GPU (batch_size=256 je bezpečný pro 12GB+ VRAM)
embeddings = model.encode(
    texts, 
    batch_size=256, 
    show_progress_bar=True, 
    normalize_embeddings=True,
    convert_to_numpy=True
)

print("4. Ukládám vektory do souboru wiki_embeddings.npy...")
np.save("wiki_embeddings.npy", embeddings.astype(np.float32))

print("HOTOVO! Soubor wiki_embeddings.npy byl úspěšně vytvořen.")
