# DATA EXPLORATION

# use ijson parser for large json file
import ijson 
from collections import Counter

 # file with just text/author/subreddit/etc (no /s markers)
comments = "comments.json"

# counting sarcastic & not (data not available in this file)
sarcastic_count = 0
non_sarcastic_count = 0

# coutning top authors & subreddits
authors = []
subreddits = []

# json file is structured as a dictionary - go through key value pairs
with open(comments, "rb") as f:
    for key, value in ijson.kvitems(f, ""):
        # check if comment has sarcasm indicator
        text = value.get("text", "")
        if "/s" in text:
            sarcastic_count += 1
        else:
            non_sarcastic_count += 1

        authors.append(value.get("author"))
        subreddits.append(value.get("subreddit"))

print("Sarcastic comments:", sarcastic_count) # 0
print("Non-sarcastic comments:", non_sarcastic_count) # 12,704,751

# authors
author_counts = Counter(authors).most_common(10)
print("Top 10 authors:", author_counts)

# subreddits
subreddit_counts = Counter(subreddits).most_common(10)
print("Top 10 subreddits:", subreddit_counts)
