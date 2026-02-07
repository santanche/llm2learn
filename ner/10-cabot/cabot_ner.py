import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from collections import defaultdict

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "samrawal/bert-base-uncased_clinical-ner"
INPUT_CSV = "../../data/case-teaching-cabot/case-teaching-cabot.csv"
OUTPUT_CSV = "../../data/case-teaching-cabot/case-teaching-cabot-with-ner.csv"

# Fixed global order of labels
LABEL_ORDER = [
    "problem",
    "treatment",
    "test"
]

# -----------------------------
# Load model and tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

device = 0 if torch.cuda.is_available() else -1

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=device
)

# -----------------------------
# Subword merging
# -----------------------------
def merge_subwords(entities):
    """
    Merge WordPiece subwords (##...) into full tokens,
    even if labels disagree. Assign a dominant label.
    """
    if not entities:
        return []

    merged = []
    i = 0

    while i < len(entities):
        current = entities[i].copy()
        word = current["text"]
        start = current["start"]
        end = current["end"]

        labels = [current["label"]]
        scores = [current["score"]]

        j = i + 1
        while j < len(entities):
            next_ent = entities[j]

            if next_ent["text"].startswith("##"):
                word += next_ent["text"][2:]
                end = next_ent["end"]
                labels.append(next_ent["label"])
                scores.append(next_ent["score"])
                j += 1
            else:
                break

        # Choose dominant label (highest mean score)
        label_scores = {}
        for lbl, sc in zip(labels, scores):
            label_scores.setdefault(lbl, []).append(sc)

        dominant_label = max(
            label_scores.items(),
            key=lambda x: sum(x[1]) / len(x[1])
        )[0]

        merged.append({
            "text": word,
            "label": dominant_label,
            "score": round(sum(scores) / len(scores), 4),
            "start": start,
            "end": end
        })

        i = j

    return merged

def clean_entities(entities):
    """
    Remove from the entity text punctuation, delimiters, brackets/parentheses and quotation marks.
    Then remove entities of size 1 that are likely artifacts of tokenization.
    """
    cleaned = []
    for ent in entities:
        text = ''.join(c for c in ent["text"] if c not in ".,;:()[]{}\"'`“”‘’<>").strip()
        if len(text) > 1:
            ent["text"] = text
            cleaned.append(ent)
    return cleaned

# -----------------------------
# NER extraction
# -----------------------------
def extract_ner_entities(text):
    if not isinstance(text, str) or text.strip() == "":
        return []

    raw_entities = ner_pipeline(text)

    entities = [
        {
            "text": ent["word"].strip(),
            "label": ent["entity_group"],
            "score": round(ent["score"], 4),
            "start": ent["start"],
            "end": ent["end"]
        }
        for ent in raw_entities
    ]

    return clean_entities(merge_subwords(entities))

# -----------------------------
# Summary construction
# -----------------------------
def build_ner_summary(entities):
    """
    Builds a compact summary:
    [LABEL1]:fragment1;fragment2.[LABEL2]:fragment1;fragment2
    """
    label_to_fragments = defaultdict(set)

    for ent in entities:
        label_to_fragments[ent["label"]].add(ent["text"])

    summary_parts = []

    for label in LABEL_ORDER:
        if label in label_to_fragments:
            fragments = sorted(label_to_fragments[label])
            summary_parts.append(f"[{label.upper()}]:{';'.join(fragments)}")

    return ".".join(summary_parts)

# # -----------------------------
# # Load CSV
# # -----------------------------
# df = pd.read_csv(INPUT_CSV)

# if "case" not in df.columns:
#     raise ValueError("CSV must contain a column named 'case'")

# # -----------------------------
# # Apply NER
# # -----------------------------
# df["ner"] = df["case"].apply(extract_ner_entities)
# df["ner_summary"] = df["ner"].apply(build_ner_summary)

# # -----------------------------
# # Save result
# # -----------------------------
# df.to_csv(
#     OUTPUT_CSV,
#     index=False,
#     encoding="utf-8",
#     quoting=1  # csv.QUOTE_ALL (safe for JSON-like columns)
# )

# print(f"NER completed. Output saved to {OUTPUT_CSV}")
