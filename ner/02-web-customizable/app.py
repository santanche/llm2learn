from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    pipeline,
    BertTokenizerFast,
    BertModel,
)
import gensim.downloader as api
import nltk
from nltk.tokenize import word_tokenize

# -------------------------------------------------
# Initialization
# -------------------------------------------------
app = FastAPI(title="Clinical NER Comparison Demo")

app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------------------------
# Models loaded ONCE (important)
# -------------------------------------------------
clinical_ner = pipeline(
    "token-classification",
    model="samrawal/bert-base-uncased_clinical-ner",
    aggregation_strategy="simple"
)

bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

w2v = api.load("word2vec-google-news-300")

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# -------------------------------------------------
# API schema
# -------------------------------------------------
class NERRequest(BaseModel):
    text: str
    prototypes: list[str]

# -------------------------------------------------
# Utility functions
# -------------------------------------------------
def build_prototypes_bert(words, embeddings, word_ids, prototype_words):
    prototypes = {}
    for pw in prototype_words:
        idxs = [
            i for i, wid in enumerate(word_ids)
            if wid is not None and words[wid] == pw
        ]
        if idxs:
            prototypes[pw] = embeddings[idxs].mean(axis=0)
    return prototypes


def bert_similarity_ner(text, prototype_words):
    words = text.lower().split()

    encoded = bert_tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        return_offsets_mapping=True
    )

    encoded.pop("offset_mapping")

    with torch.no_grad():
        outputs = bert_model(**encoded)

    embeddings = outputs.last_hidden_state.squeeze(0).numpy()
    tokens = bert_tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    word_ids = encoded.word_ids()

    prototypes = build_prototypes_bert(words, embeddings, word_ids, prototype_words)

    results = []
    for token, emb, wid in zip(tokens, embeddings, word_ids):
        if wid is None or token.startswith("##"):
            continue

        sims = {
            pw: cosine_similarity(
                emb.reshape(1, -1),
                proto.reshape(1, -1)
            )[0][0]
            for pw, proto in prototypes.items()
        }

        if sims:
            best = max(sims, key=sims.get)
            if sims[best] > 0.75:
                results.append({
                    "text": token,
                    "label": best,
                    "score": float(sims[best])
                })

    return results


def w2v_similarity_ner(text, prototype_words):
    tokens = word_tokenize(text.lower())

    results = []
    for t in tokens:
        if t in w2v:
            sims = {
                pw: cosine_similarity(
                    w2v[t].reshape(1, -1),
                    w2v[pw].reshape(1, -1)
                )[0][0]
                for pw in prototype_words if pw in w2v
            }
            if sims:
                best = max(sims, key=sims.get)
                if sims[best] > 0.65:
                    results.append({
                        "text": t,
                        "label": best,
                        "score": float(sims[best])
                    })
    return results

def make_json_safe(obj):
    """
    Recursively convert NumPy types to native Python types
    so FastAPI can serialize them.
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    with open("static/index.html") as f:
        return f.read()


@app.post("/run")
def run_ner(req: NERRequest):
    text = req.text
    prototype_words = [p.strip().lower() for p in req.prototypes if p.strip()]

    log = []

    log.append("Running Pipeline 1 (fine-tuned clinical BERT)")
    p1 = make_json_safe(clinical_ner(text))

    log.append("Running Pipeline 2 (vanilla BERT + similarity)")
    p2 = make_json_safe(bert_similarity_ner(text, prototype_words))

    log.append("Running Pipeline 3 (Word2Vec + similarity)")
    p3 = make_json_safe(w2v_similarity_ner(text, prototype_words))

    return {
        "pipeline_1": p1,
        "pipeline_2": p2,
        "pipeline_3": p3,
        "log": log
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)