# Converts SARC’s paired format into a flat CSV.
# For each line in the dataset, it extracts the main comment plus its two replies,
# then writes two rows—one per reply—with:
# response_id, response_text, main_comment_id, main_comment_text, label.
# *used partial help from Ai to help with formatting and debugging, as we were not familiar with json/csv semantics.
import json
import csv

# run this file on both train-balanced and test-balanced to get train and test data 
DATA_PATH = "data/test-balanced.csv" # this can be replaced by desired file 
COMMENTS_PATH = "data/test_small_comments_fixed.json"
OUT_PATH = "test_new_responses_flat.csv"

# Load comments
with open(COMMENTS_PATH, "r", encoding="utf-8") as f:
    comments = json.load(f)
    
# print how many comments were loaded from JSON
print("Loaded comments:", len(comments))

# later used for checking if our program wrote to the file without any errors 
lines_total = 0
lines_used = 0
lines_skipped = 0
rows_written = 0

# open the input CSV-like file and the output CSV at the same time
with open(DATA_PATH, encoding="utf-8") as fin, \
     open(OUT_PATH, "w", encoding="utf-8", newline="") as fout:

    writer = csv.writer(fout)    # CSV writer object to write rows into fout
    writer.writerow(["response_id", "response_text",    # header row for the output CSV
                     "main_comment_id", "main_comment_text", "label"])
         
    # iterate over each line (string) in the input file
    for raw in fin:
        line = raw.strip()    # strip leading/trailing whitespace and newline
        if not line:            # if line is empty after stripping, skip it
            continue

        lines_total += 1    # count this as a line we've processed

        if line.count("|") < 2:   # each valid line should have at least 2 '|' characters: main | replies | labels
            lines_skipped += 1    # line format is wrong → skip
            continue

        # main comment and replies
        left, right = line.split("|", 1)    # split into left side (main id) and everything after first '|'
        if "|" not in right:                # ensure there's another '|' to separate replies and labels
            lines_skipped += 1
            continue

        main_comment_id = left.strip()

        # only continue if main comment exists in our comments dict
        if main_comment_id not in comments:
            lines_skipped += 1    # skip if we don't have the text
            continue

        # get the main comment's text from JSON
        main_comment_text = comments[main_comment_id]["text"]    

        # split remaining part into replies segment and labels segment
        replies_part, labels_part = right.split("|", 1)

        # replies_part looks like: "reply_id1 reply_id2"; split on whitespace → list of reply IDs
        reply_ids = replies_part.strip().split()

        # labels_part looks like: "label1 label2"; split on whitespace → list of labels (e.g. "0 1")
        labels = labels_part.strip().split()

        # dataset requires exactly 2 replies and 2 labels
        if len(reply_ids) != 2 or len(labels) != 2:
            lines_skipped += 1    # if it’s not exactly 2 and 2, line doesn’t match expected format
            continue

        rid1, rid2 = reply_ids    # unpack the two reply IDs
        lab1, lab2 = labels       # unpack the two labels

        # only keep if both replies exist in the comments dict
        if rid1 not in comments or rid2 not in comments:
            lines_skipped += 1    # if we don't have text for one of the replies, skip
            continue

        # Write rows
        # first reply row - (one row per reply – this is the "flattening")
        writer.writerow([rid1,
                         comments[rid1]["text"],
                         main_comment_id,
                         main_comment_text,
                         lab1])
        
        # second reply row
        writer.writerow([rid2,
                         comments[rid2]["text"],
                         main_comment_id,
                         main_comment_text,
                         lab2])

        rows_written += 2    # we wrote two rows for this line
        lines_used += 1      # this input line was successfully used

# after processing all lines, print stats
print("lines_total:", lines_total)
print("lines_used:", lines_used)
print("lines_skipped:", lines_skipped)
print("rows_written:", rows_written)
print("expected rows:", lines_used * 2)
