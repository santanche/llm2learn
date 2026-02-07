import pandas as pd
import re
from difflib import SequenceMatcher

# ---------- Configuration ----------
INPUT_CSV = "../../data/case-teaching-cabot/case-teaching-cabot.csv"
OUTPUT_CSV = "../../data/case-teaching-cabot/case-teaching-cabot-with-diagnosis.csv"

TARGET_WORD = "diagnosis"

NEAR_THRESHOLD = 0.75
FAR_THRESHOLD = 0.55

SEPARATORS = r"[:;.,·•]"

# ---------- OCR normalization ----------
def ocr_normalize(word):
    """
    Normalize common OCR confusions conservatively.
    """
    word = word.lower()
    word = word.replace("ff", "ss")   # long-s OCR issue
    word = word.replace("fl", "si")
    word = word.replace("fi", "si")
    word = word.replace("1", "l")
    word = word.replace("0", "o")
    return word

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# ---------- Helpers ----------
def extract_paragraph(text, start_pos):
    remainder = text[start_pos:]
    split = re.split(r"\n\s*\n", remainder, maxsplit=1)
    return split[0].strip()

def normalize(text):
    return re.sub(r"\s+", " ", text.strip())

# ---------- Diagnosis extractor ----------
def extract_diagnosis(text):
    if not isinstance(text, str):
        return ""

    # 1) Exact patterns
    exact_patterns = [
        rf"^Diagnosis{SEPARATORS}\s*(.*?)(?:\n\s*\n|\Z)",
        r"^The Diagnosis is\s*(.*?)(?:\n\s*\n|\Z)"
    ]

    for pat in exact_patterns:
        m = re.search(pat, text, re.MULTILINE | re.DOTALL)
        if m:
            return normalize(m.group(1))

    # Collect header-like candidates
    candidates = []
    for m in re.finditer(
        rf"^([A-Za-z]{{5,}})\s*{SEPARATORS}\s*",
        text,
        re.MULTILINE
    ):
        word = m.group(1)
        candidates.append((word, m.end()))

    # 2) Near similarity pass
    for word, pos in candidates:
        score = similarity(ocr_normalize(word), TARGET_WORD)
        if score >= NEAR_THRESHOLD:
            return normalize(extract_paragraph(text, pos))

    # 3) Distant similarity fallback (only if nothing found)
    for word, pos in candidates:
        score = similarity(ocr_normalize(word), TARGET_WORD)
        if score >= FAR_THRESHOLD and abs(len(word) - len(TARGET_WORD)) <= 3:
            return normalize(extract_paragraph(text, pos))

    # 4) Sentence form fallback: "The <word> is"
    for m in re.finditer(
        r"^The\s+([A-Za-z]{5,})\s+is\s+",
        text,
        re.MULTILINE
    ):
        word = m.group(1)
        score = similarity(ocr_normalize(word), TARGET_WORD)
        if score >= FAR_THRESHOLD:
            return normalize(extract_paragraph(text, m.end()))

    return ""

# ---------- Main pipeline ----------
# def main():
#     df = pd.read_csv(INPUT_CSV)

#     if not {"number", "case"}.issubset(df.columns):
#         raise ValueError("CSV must contain 'number' and 'case' columns")

#     df_out = pd.DataFrame({
#         "number": df["number"],
#         "diagnosis": df["case"].apply(extract_diagnosis)
#     })

#     df_out.to_csv(OUTPUT_CSV, index=False)
#     print(f"Saved {len(df_out)} rows to {OUTPUT_CSV}")

# ---------- Run ----------
# if __name__ == "__main__":
#     main()
