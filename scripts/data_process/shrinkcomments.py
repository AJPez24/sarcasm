# Creating a smaller comments.json file with only comments that match IDs in test-balanced.csv
import csv
import ijson
import json

TRAIN_PATH = "data/test-balanced.csv"
BIG_COMMENTS_PATH = "data/comments.json"
OUT_SMALL = "test_small_comments_fixed.json"


# Collect all necessary IDs
# (Main comment ID to the left side of each line, both reply IDs to the right side of each line)
needed_ids = set()

# Go through each line in the csv file
with open(TRAIN_PATH, encoding="utf-8") as f:
    # Skip empty lines
    for raw in f:
        line = raw.strip()
        if not line:
            continue
        if "|" not in line:
            continue

        # Split top-level comment from reply block
        main_comment_id, right = line.split("|", 1)
        main_comment_id = main_comment_id.strip()

        # ALWAYS keep the main comment ID
        needed_ids.add(main_comment_id)

        if "|" not in right:
            continue

        # Split replies from labels
        replies_part, labels_part = right.split("|", 1)

        # Replies are space separated
        reply_ids = replies_part.strip().split()
        # Add reply IDs
        for rid in reply_ids:
            needed_ids.add(rid)

print("Total IDs needed:", len(needed_ids))


# Stream the big JSON and extract only the IDs we need
# Help from ChatGPT to understand ijson to deal with large json files

# Dictionary to hold necessary comments
small_comments = {}

with open(BIG_COMMENTS_PATH, "rb") as f:
    # JSON is set up as a dictionary - stream the key value pairs
    for cid, obj in ijson.kvitems(f, ""):
        # Check if ID matches IDs found in CSV
        if cid in needed_ids:
            small_comments[cid] = obj

print("Found in big comments:", len(small_comments))

# See if any comment IDs in the CSV were not included in the JSON
missing = needed_ids - small_comments.keys()
print("Missing IDs:", len(missing))
if missing:
    print("Example missing IDs:", list(missing)[:20])


# Save the new smaller JSON file
with open(OUT_SMALL, "w", encoding="utf-8") as f:
    json.dump(small_comments, f)

print("Wrote:", OUT_SMALL)
