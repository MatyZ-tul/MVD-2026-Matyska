import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# --- Kontrola GPU a výpočetní architektury (sm_...) ---
if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    sm_version = f"sm_{major}{minor}"
    
    print(f"✅ GPU Nalezeno: {gpu_name}")
    print(f"🚀 Používaná architektura (Compute Capability): {sm_version} (CUDA verze: {torch.version.cuda})")
else:
    device = "cpu"
    print("⚠️ GPU nenalezeno, kód poběží na CPU.")

# 1. Načtení Wikipedie (pro přístup k názvům a textům článků)
dataset = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")

# 2. Načtení uložených vektorů (nyní už soubor existuje)
emb = np.load("wiki_embeddings.npy").astype(np.float32) # převedeme zpět na float32 pro výpočet

# 3. Načtení modelu s explicitním určením zařízení (využije GPU, pokud je k dispozici)
model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", device=device)

# 4. Funkce pro neuronové vyhledávání
def search_neural(query, top_k=5):
    # Převod dotazu na vektor (proběhne na GPU, pokud je model na CUDA)
    q_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]

    # Skóre je maticové násobení (NumPy běží na CPU, ale pro malé vektory je to okamžité)
    scores = emb @ q_vec

    # Seřazení od nejvyššího po nejnižší
    best_indices = scores.argsort()[::-1][:top_k]

    print(f"\n--- Výsledky pro dotaz: '{query}' ---")
    for idx in best_indices:
        title = dataset[int(idx)]['title']
        score = scores[idx]
        print(f"Skóre: {score:.4f} | Název článku: {title}")

# 5. Testování na doporučených dotazech ze zadání:
search_neural("how do plants make food from sunlight")  # Mělo by najít Photosynthesis
search_neural("largest animal that ever lived")        # Mělo by najít Blue whale
search_neural("coursera vs udacity machine learning")
