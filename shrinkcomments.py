import csv
import ijson
import json

TRAIN_PATH = "data/test-balanced.csv"
BIG_COMMENTS_PATH = "data/comments.json"        # <-- full 2.6GB file
OUT_SMALL = "small_comments_fixed.json"    # <-- new fixed small file

# ------------------------------------------------------------
# 1) Collect ALL reply IDs from train-balanced
# ------------------------------------------------------------
needed_ids = set()

with open(TRAIN_PATH, encoding="utf-8") as f:
    for raw in f:
        line = raw.strip()
        if not line:
            continue
        if "|" not in line:
            continue

        # split on first pipe
        left, right = line.split("|", 1)
        if "|" not in right:
            continue

        replies_part, labels_part = right.split("|", 1)

        # replies are space-separated
        reply_ids = replies_part.strip().split()
        for rid in reply_ids:
            needed_ids.add(rid)

print("Reply IDs needed from test-balanced:", len(needed_ids))

# ------------------------------------------------------------
# 2) Stream the BIG comments.json and keep only those IDs
#    comments.json is a SINGLE BIG JSON OBJECT: {id: {...}, id2: {...}, ...}
# ------------------------------------------------------------
small_comments = {}

with open(BIG_COMMENTS_PATH, "rb") as f:
    for cid, obj in ijson.kvitems(f, ""):
        if cid in needed_ids:
            small_comments[cid] = obj

print("Found in big comments:", len(small_comments))

missing = needed_ids - small_comments.keys()
print("Missing reply IDs:", len(missing))
if missing:
    print("Example missing IDs:", list(missing)[:20])

# ------------------------------------------------------------
# 3) Save the fixed small comments file
# ------------------------------------------------------------
with open(OUT_SMALL, "w", encoding="utf-8") as f:
    json.dump(small_comments, f)

print("Wrote fixed small comments to", OUT_SMALL)
