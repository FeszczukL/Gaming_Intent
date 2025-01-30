import pandas as pd
from transformers import pipeline


file_path = "reddit_gaming_sample.csv"
df = pd.read_csv(file_path)

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["gaming-related", "not gaming-related"]

def classify_comment(comment):
    result = classifier(comment, candidate_labels)
    return 1 if result["labels"][0] == "gaming-related" else 0


df["gaming_related"] = df["Comment"].apply(classify_comment)
df.to_csv("reddit_gaming_sample_labeled.csv", index=False)
