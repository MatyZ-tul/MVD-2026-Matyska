import gzip
import json
import torch
import numpy as np
from transformers import AutoModel, AutoProcessor

print("1. Načítám dataset curlie_cs.jsonl.gz...")
with gzip.open("curlie_cs.jsonl.gz", "rt", encoding="utf-8") as f:
    pages = [json.loads(line) for line in f]

# Seřazení stránek podle délky HTML řetězce (důležité pro efektivitu batchování)
pages = sorted(pages, key=lambda p: len(p["html"]))

print(f"Dataset načten, celkem stránek: {len(pages)}")

print("2. Inicializuji HTML-LM model...")
processor = AutoProcessor.from_pretrained("Seznam/html-lm", trust_remote_code=True)
model = AutoModel.from_pretrained("Seznam/html-lm").to("cuda").eval()

batch_size = 16
all_vectors = []
all_splits = []
all_categories = []

print("3. Zpracovávám stránky v dávkách na GPU...")
for i in range(0, len(pages), batch_size):
    batch = pages[i:i+batch_size]
    html_texts = [p["html"] for p in batch]
    
    # Tokenizace a zarovnání délek
    ids = processor(html_texts, truncation=True, max_length=4096)["input_ids"]
    x = processor.tokenizer.pad({"input_ids": ids}, return_tensors="pt", padding=True).to("cuda")
    
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).last_hidden_state
        
    # Získání [CLS] vektoru pro každou stránku v dávce (rozměr [batch_size, 768])
    vectors = out[:, 0].float().cpu().numpy()
    all_vectors.append(vectors)
    
    for p in batch:
        all_splits.append(p["split"])
        all_categories.append(p["category"])
        
    if (i // batch_size) % 50 == 0:
        print(f"Zpracováno {i} / {len(pages)} stránek...")

# Spojení a uložení výsledků
X = np.concatenate(all_vectors, axis=0)
np.save("html_lm_vectors.npy", X)
np.save("splits.npy", np.array(all_splits))
np.save("categories.npy", np.array(all_categories))

print("HOTOVO! Vektory a metadata úspěšně uložena do .npy souborů.")

