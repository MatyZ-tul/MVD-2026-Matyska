import re
import pickle
import spacy
from datasets import load_dataset
from collections import defaultdict, Counter

# Načtení datasetu
wiki = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")

titles = [item["title"] for item in wiki]
texts = [item["text"] for item in wiki]

# Načtení spaCy modelu (vypneme parser a ner pro rychlost)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess_batch(docs_list, n_proc):
    processed = []
    # nlp.pipe efektivně zpracovává dávky textu paralelně pomocí n_process
    for doc in nlp.pipe(docs_list, batch_size=500, n_process=n_proc):
        # Vytažení lemmat přímo z již zpracovaného spaCy doc objektu
        # Rovnou ošetříme interpunkci / ne-alfa znaky a mezery
        lemmas = " ".join([token.lemma_.lower() for token in doc if not token.is_space and not token.is_punct])
        processed.append(lemmas)
    return processed

print("Zpracovávám názvy (titles)...")
processed_titles = preprocess_batch(titles, n_proc=16)

print("Zpracovávám texty (texts)...")
processed_texts = preprocess_batch(texts, n_proc=16)

def build_inverted_index_with_tf(processed_docs):
    index = defaultdict(dict)
    for doc_id, doc in enumerate(processed_docs):
        words = doc.split()
        tf_counts = Counter(words)
        for word, count in tf_counts.items():
            index[word][doc_id] = count
    return index

print("Stavím invertovaný index pro názvy...")
index_title = build_inverted_index_with_tf(processed_titles)

print("Stavím invertovaný index pro texty...")
index_text = build_inverted_index_with_tf(processed_texts)

print("Ukládám soubory...")
with open("index_title.pkl", "wb") as f:
    pickle.dump(index_title, f)

with open("index_text.pkl", "wb") as f:
    pickle.dump(index_text, f)

print("Hotovo!")
