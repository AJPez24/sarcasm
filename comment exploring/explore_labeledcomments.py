import json
import pandas as pd

# 1. Load the comment metadata (where the text lives)
with open("main/comments.json", "r") as f:
    comments = json.load(f)   # {comment_id: {..., "text": "...", ...}}

# 2. Load the balanced train file
df = pd.read_csv(
    "main/train_balanced.csv",
    header=None,
    sep="|",
    names=["chain", "responses", "labels"]
)

# 3. Explode each row into (response_id, label, text)
records = []

for _, row in df.iterrows():
    resp_ids = row["responses"].split()          # ["c1", "c2", ...]
    labels   = row["labels"].split()             # ["0", "1", ...]
    
    for rid, lab in zip(resp_ids, labels):
        data = comments.get(rid, {})
        text = data.get("text") or data.get("body")  # depending on field name
        
        records.append({
            "response_id": rid,
            "label": int(lab),
            "text": text
        })

responses_df = pd.DataFrame(records)
# Save to a new CSV
responses_df.to_csv("responses_labeled.csv", index=False)

