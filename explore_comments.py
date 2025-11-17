# DATA EXPLORATION

# use ijson parser for large json file
import ijson 

 # file with just comments and information (no /s markers)
comments = "comments.json"

# json file is structured as a dictionary - go through key value pairs
sarcastic_count = 0
non_sarcastic_count = 0
with open(comments, "rb") as f:
    for key, value in ijson.kvitems(f, ""):
        # check if comment has sarcasm indicator
        text = value.get("text", "")
        if "/s" in text:
            sarcastic_count += 1
        else:
            non_sarcastic_count += 1

# 12,704,751 total, none marked as sarcastic
print("Sarcastic comments:", sarcastic_count)
print("Non-sarcastic comments:", non_sarcastic_count)