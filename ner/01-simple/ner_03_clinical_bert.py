from transformers import pipeline

text = "The patient was diagnosed with diabetes and prescribed metformin."

clinical_ner = pipeline(
    task="token-classification",
    model="samrawal/bert-base-uncased_clinical-ner",
    aggregation_strategy="simple"
)

entities_1 = clinical_ner(text)

print("Pipeline 1 — Fine-tuned Clinical BERT")
for e in entities_1:
    print(e)
