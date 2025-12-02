import csv
import ijson
import json

TRAIN_PATH = "data/test-balanced.csv"
BIG_COMMENTS_PATH = "data/comments.json"
OUT_SMALL = "test_small_comments_fixed.json"

# ------------------------------------------------------------
# 1) Collect ALL IDs we need:
#    - Main comment ID (left side of each line)
#    - Both reply IDs (right side of each line)
# ------------------------------------------------------------
needed_ids = set()

with open(TRAIN_PATH, encoding="utf-8") as f:
    for raw in f:
        line = raw.strip()
        if not line:
            continue
        if "|" not in line:
            continue

        # split top-level comment from reply block
        main_comment_id, right = line.split("|", 1)
        main_comment_id = main_comment_id.strip()

        # ALWAYS keep the main comment ID
        needed_ids.add(main_comment_id)

        if "|" not in right:
            continue

        replies_part, labels_part = right.split("|", 1)

        # replies are space separated
        reply_ids = replies_part.strip().split()
        for rid in reply_ids:
            needed_ids.add(rid)

print("Total IDs needed:", len(needed_ids))

# ------------------------------------------------------------
# 2) Stream the big JSON and extract only the IDs we need
# ------------------------------------------------------------
small_comments = {}

with open(BIG_COMMENTS_PATH, "rb") as f:
    for cid, obj in ijson.kvitems(f, ""):
        if cid in needed_ids:
            small_comments[cid] = obj

print("Found in big comments:", len(small_comments))

missing = needed_ids - small_comments.keys()
print("Missing IDs:", len(missing))
if missing:
    print("Example missing IDs:", list(missing)[:20])

# ------------------------------------------------------------
# 3) Save the small JSON file
# ------------------------------------------------------------
with open(OUT_SMALL, "w", encoding="utf-8") as f:
    json.dump(small_comments, f)

print("Wrote:", OUT_SMALL)
