import ijson

needed_ids = set()

# 1) Collect all IDs from train-balanced
with open("data/train-balanced.csv", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        tokens = line.split()

        for tok in tokens:
            # if it's like "c07fjge|1"
            if '|' in tok:
                id_part = tok.split('|', 1)[0]
                if id_part and not id_part.isdigit():
                    needed_ids.add(id_part)
            else:
                # could be an ID (e.g. "7zp51", "c07uja7"), ignore pure labels
                if tok not in {"0", "1"} and not tok.isdigit():
                    needed_ids.add(tok)

print("IDs needed from train-balanced:", len(needed_ids))

# 2) Stream comments.json and keep only what we need
small_comments = {}

with open("data/comments.json", "rb") as f:
    # top-level object: { "id": {...}, "id2": {...}, ... }
    for cid, obj in ijson.kvitems(f, ""):
        if cid in needed_ids:
            small_comments[cid] = obj

print("Have texts for:", len(small_comments), "IDs")

missing = needed_ids - small_comments.keys()
print("Missing IDs:", len(missing))
# optional peek:
print("Example missing:", list(missing)[:20])
