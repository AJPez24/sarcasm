# data visualizations

import matplotlib.pyplot as plt


# use ijson parser for large json file
import ijson 
from collections import Counter

 # file with just text/author/subreddit/etc (no /s markers)
comments = "data/comments.json"

# coutning top authors & subreddits
authors = []
subreddits = []

# json file is structured as a dictionary - go through key value pairs
with open(comments, "rb") as f:
    for key, value in ijson.kvitems(f, ""):

        authors.append(value.get("author"))
        subreddits.append(value.get("subreddit"))

# authors
author_counts = Counter(authors).most_common(10)

# Filter out authors named "[deleted]" - bots
filtered_author_counts = [(a, c) for a, c in author_counts if a != "[deleted]"]

authors_list = [a for a, c in filtered_author_counts]
author_values = [c for a, c in filtered_author_counts]

plt.figure(figsize=(10, 5))
plt.bar(authors_list, author_values)
plt.xticks(rotation=45, ha='right')
plt.title("Top 10 Authors by Number of Comments")
plt.xlabel("Author")
plt.ylabel("Number of Comments")
plt.tight_layout()
plt.show()

# subreddits
subreddit_counts = Counter(subreddits).most_common(10)

subreddit_list = [s for s, c in subreddit_counts]
subreddit_values = [c for s, c in subreddit_counts]

plt.figure(figsize=(10, 5))
plt.bar(subreddit_list, subreddit_values)
plt.xticks(rotation=45, ha='right')
plt.title("Top 10 Subreddits by Number of Comments")
plt.xlabel("Subreddit")
plt.ylabel("Number of Comments")
plt.tight_layout()
plt.show()