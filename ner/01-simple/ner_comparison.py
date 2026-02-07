"""
Comparison of three clinical NER pipelines:

1) Fine-tuned clinical BERT (real NER)
2) Vanilla BERT embeddings + heuristic classifier (no fine-tuning)
3) Static Word2Vec embeddings + heuristic classifier
"""

# -----------------------------
# Common imports
# -----------------------------
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Example clinical text
# -----------------------------
# text = "The patient was diagnosed with diabetes mellitus and prescribed metformin 500 mg daily."
# text = "The patient has a cold and mild fever."
text = "The room is colder overnight."

print("TEXT:", text)

# ============================================================
# PIPELINE 1 — Fine-tuned Clinical BERT NER
# ============================================================
from transformers import pipeline

clinical_ner = pipeline(
    task="token-classification",
    model="samrawal/bert-base-uncased_clinical-ner",
    aggregation_strategy="simple"
)

print("\nPIPELINE 1 — Fine-tuned Clinical BERT")
entities_1 = clinical_ner(text)
for e in entities_1:
    print(e)

# ============================================================
# PIPELINE 2 — Vanilla BERT embeddings (NO fine-tuning)
# ============================================================

from transformers import BertTokenizerFast, BertModel

# ------------------------------------------------------------
# 1) Pre-tokenize text into words
# ------------------------------------------------------------
words = text.lower().split()

# ------------------------------------------------------------
# 2) Load tokenizer and model (FAST tokenizer is mandatory)
# ------------------------------------------------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# ------------------------------------------------------------
# 3) Tokenize WITH word alignment
# ------------------------------------------------------------
encoded = tokenizer(
    words,
    is_split_into_words=True,
    return_tensors="pt",
    return_offsets_mapping=True
)

# Remove offsets before passing to the model
offset_mapping = encoded.pop("offset_mapping")

# ------------------------------------------------------------
# 4) Forward pass
# ------------------------------------------------------------
with torch.no_grad():
    outputs = model(**encoded)

# ------------------------------------------------------------
# 5) Extract embeddings and alignment info
# ------------------------------------------------------------
embeddings = outputs.last_hidden_state.squeeze(0).numpy()
token_strings = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
word_ids = encoded.word_ids()

# ------------------------------------------------------------
# 6) Build prototype vectors safely
# ------------------------------------------------------------
def build_prototype(target_word, words, embeddings, word_ids):
    """
    Average embeddings of all sub-tokens that correspond
    to the given target word.
    """
    idxs = [
        i for i, wid in enumerate(word_ids)
        if wid is not None and words[wid] == target_word
    ]
    if not idxs:
        return None
    return embeddings[idxs].mean(axis=0)

label_prototypes = {
    "COLD": build_prototype("cold", words, embeddings, word_ids),
    "DISEASE": build_prototype("diabetes", words, embeddings, word_ids),
    "DRUG": build_prototype("metformin", words, embeddings, word_ids),
}

# Remove missing prototypes (defensive programming)
label_prototypes = {
    k: v for k, v in label_prototypes.items() if v is not None
}

# ------------------------------------------------------------
# 7) Heuristic entity recognition via cosine similarity
# ------------------------------------------------------------
print("\nPIPELINE 2 — Vanilla BERT + heuristic similarity")

for token, emb, wid in zip(token_strings, embeddings, word_ids):
    # Skip special tokens and subword continuations
    if wid is None or token.startswith("##"):
        continue

    sims = {
        label: cosine_similarity(
            emb.reshape(1, -1),
            proto.reshape(1, -1)
        )[0][0]
        for label, proto in label_prototypes.items()
    }

    if sims:
        label = max(sims, key=sims.get)
        if sims[label] > 0.75:
            print(token, "→", label, f"(sim={sims[label]:.2f})")

# ============================================================
# PIPELINE 3 — Static Word2Vec embeddings
# ============================================================
import gensim.downloader as api
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt", quiet=True)
nltk.download('punkt_tab')

w2v = api.load("word2vec-google-news-300")

tokens = word_tokenize(text.lower())

vectors = []
valid_tokens = []

for t in tokens:
    if t in w2v:
        vectors.append(w2v[t])
        valid_tokens.append(t)

prototypes_w2v = {
    "COLD": w2v["cold"],
    "DISEASE": w2v["diabetes"],
    "DRUG": w2v["metformin"],
}

print("\nPIPELINE 3 — Word2Vec + heuristic similarity")
for token, vec in zip(valid_tokens, vectors):
    sims = {
        label: cosine_similarity(
            vec.reshape(1, -1),
            proto.reshape(1, -1)
        )[0][0]
        for label, proto in prototypes_w2v.items()
    }
    label = max(sims, key=sims.get)
    if sims[label] > 0.65:
        print(token, "→", label, f"(sim={sims[label]:.2f})")
