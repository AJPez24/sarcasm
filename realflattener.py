import json
import csv

TRAIN_PATH = "data/test-balanced.csv"
COMMENTS_PATH = "data/test_small_comments_fixed.json"
OUT_PATH = "test_new_responses_flat.csv"

# Load comments
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
    writer.writerow(["response_id", "response_text",
                     "main_comment_id", "main_comment_text", "label"])

    for raw in fin:
        line = raw.strip()
        if not line:
            continue

        lines_total += 1

        if line.count("|") < 2:
            lines_skipped += 1
            continue

        # main comment and replies
        left, right = line.split("|", 1)
        if "|" not in right:
            lines_skipped += 1
            continue

        main_comment_id = left.strip()

        # only continue if main comment exists
        if main_comment_id not in comments:
            lines_skipped += 1
            continue

        main_comment_text = comments[main_comment_id]["text"]

        replies_part, labels_part = right.split("|", 1)
        reply_ids = replies_part.strip().split()
        labels = labels_part.strip().split()

        # dataset requires exactly 2 replies and 2 labels
        if len(reply_ids) != 2 or len(labels) != 2:
            lines_skipped += 1
            continue

        rid1, rid2 = reply_ids
        lab1, lab2 = labels

        # only keep if both replies exist
        if rid1 not in comments or rid2 not in comments:
            lines_skipped += 1
            continue

        # Write rows
        writer.writerow([rid1,
                         comments[rid1]["text"],
                         main_comment_id,
                         main_comment_text,
                         lab1])

        writer.writerow([rid2,
                         comments[rid2]["text"],
                         main_comment_id,
                         main_comment_text,
                         lab2])

        rows_written += 2
        lines_used += 1

print("lines_total:", lines_total)
print("lines_used:", lines_used)
print("lines_skipped:", lines_skipped)
print("rows_written:", rows_written)
print("expected rows:", lines_used * 2)
