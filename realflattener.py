import json
import csv

TRAIN_PATH = "data/test-balanced.csv"
COMMENTS_PATH = "data/small_comments_fixed_test.json"   # <-- use the NEW file
OUT_PATH = "responses_flat_test.csv"

# 1) Load comments (single big dict)
with open(COMMENTS_PATH, "r", encoding="utf-8") as f:
    comments = json.load(f)

print("Loaded comments:", len(comments))

lines_total = 0
lines_used = 0
lines_skipped = 0
rows_written = 0

with open(TRAIN_PATH, encoding="utf-8") as fin, \
     open(OUT_PATH, "w", encoding="utf-8", newline="") as fout:

    writer = csv.writer(fout)
    writer.writerow(["response_id", "response_text", "label"])

    for raw in fin:
        line = raw.strip()
        if not line:
            continue

        lines_total += 1

        if line.count("|") < 2:
            lines_skipped += 1
            continue

        left, right = line.split("|", 1)
        if "|" not in right:
            lines_skipped += 1
            continue

        replies_part, labels_part = right.split("|", 1)

        reply_ids = replies_part.strip().split()
        labels = labels_part.strip().split()

        if len(reply_ids) != 2 or len(labels) != 2:
            lines_skipped += 1
            continue

        rid1, rid2 = reply_ids
        lab1, lab2 = labels

        if rid1 not in comments or rid2 not in comments:
            lines_skipped += 1
            continue

        writer.writerow([rid1, comments[rid1]["text"], lab1])
        writer.writerow([rid2, comments[rid2]["text"], lab2])
        rows_written += 2
        lines_used += 1

print("lines_total:", lines_total)
print("lines_used:", lines_used)
print("lines_skipped:", lines_skipped)
print("rows_written:", rows_written)
print("expected rows:", lines_used * 2)
