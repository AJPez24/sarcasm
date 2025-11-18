import csv
import ijson

TRAIN_PATH = "data/train-balanced.csv"   # change if needed
COMMENTS_PATH = "data/comments.json"
OUT_PATH = "data/replies_labeled.csv"

# 1) Read all response_ids and their labels from train-balanced.csv
response_labels = {}   # response_id -> label (0/1)

with open(TRAIN_PATH, newline="", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="|")
    for row in reader:
        if len(row) != 3:
            continue  # skip malformed lines
        chain, responses, labels = row
        resp_ids = responses.strip().split()
        labs = labels.strip().split()

        for rid, lab in zip(resp_ids, labs):
            response_labels[rid] = int(lab)

response_ids = set(response_labels.keys())
print(f"Collected {len(response_ids)} unique response IDs")

# 2) Stream comments.json and grab only the ones we need
id_to_text = {}  # response_id -> text

with open(COMMENTS_PATH, "r", encoding="utf-8") as f:
    parser = ijson.kvitems(f, "")  # key-value pairs at root

    for comment_id, data in parser:
        if comment_id in response_ids:
            text = data.get("text") or data.get("body") or ""
            id_to_text[comment_id] = text

print(f"Matched {len(id_to_text)} responses with text")

# 3) Write final CSV: response_id, text, label
with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["response_id", "text", "label"])

    for rid, label in response_labels.items():
        text = id_to_text.get(rid, "")
        writer.writerow([rid, text, label])

print(f"Saved labeled replies to {OUT_PATH}")
