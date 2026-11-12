import numpy as np
import torch
from datasets import load_dataset

# 1. Načtení dat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Používané zařízení: {device}")

X_np = np.load("wiki_embeddings.npy").astype(np.float32)
X = torch.tensor(X_np, device=device)

G = 40
num_samples = len(X)
epochs = 3
T = num_samples * epochs
t = 0

# 2. Inicializace vah (1600 náhodných článků)
W = X[torch.randperm(num_samples, device=device)[:G * G]].clone()

# Předpřipravení souřadnic mřížky
radek = torch.arange(G * G, device=device) // G
sloupec = torch.arange(G * G, device=device) % G

print("Začátek tréninku SOM...")
for epoch in range(epochs):
    perm = torch.randperm(num_samples, device=device)
    X_shuffled = X[perm]
    
    for i in range(num_samples):
        xj = X_shuffled[i]
        t += 1
        
        # Dynamické parametry
        alpha = 0.5 + (0.02 - 0.5) * (t / T)
        beta = alpha ** 2
        r = (G / 2) * (1 - (t / T))
        
        # Vzdálenosti ke všem neuronům (vektorizovaně)
        d2 = ((W - xj) ** 2).sum(dim=1)
        best = torch.argmin(d2)
        
        # Určení regionu
        region = ((radek - radek[best]).abs() <= r) & ((sloupec - sloupec[best]).abs() <= r)
        region[best] = False
        
        # Aktualizace vah
        W[best] += alpha * (xj - W[best])
        if region.any():
            W[region] += beta * (xj - W[region])
            
    print(f"Epocha {epoch+1}/{epochs} dokončena.")

# Uložení vah
np.save("som_W.npy", W.cpu().numpy())
print("Váhy uloženy do som_W.npy")
