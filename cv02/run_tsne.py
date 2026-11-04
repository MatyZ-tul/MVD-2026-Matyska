import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE

X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
X, y = X[:30000], y[:30000]

start_time = time.time()

tsne = TSNE(n_components=2, method="exact", random_state=42)
X_embedded = tsne.fit_transform(X)

elapsed_time = time.time() - start_time
hours = elapsed_time / 3600
print(f"Výpočet dokončen za {elapsed_time:.2f} s ({hours:.2f} h)")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_embedded[:, 0], 
    X_embedded[:, 1], 
    c=y.astype(int), 
    cmap="tab10", 
    s=1, 
    alpha=0.6
)
plt.colorbar(scatter, label="Číslice (0-9)")
plt.title(f"t-SNE na MNIST (30k vzorků, method='exact')\nČas výpočtu: {hours:.2f} h")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

plt.savefig("mnist_tsne.png", dpi=300, bbox_inches="tight")
