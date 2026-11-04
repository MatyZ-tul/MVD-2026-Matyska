import numpy as np

rng = np.random.default_rng()
A = rng.random((50000, 50000), dtype=np.float64)

print("Rozměry a typ:", A.shape, A.dtype)
print("Velikost v paměti:", A.nbytes / 1e9, "GB")
print("Průměr:", A.mean())
print("Směrodatná odchylka:", A.std())
