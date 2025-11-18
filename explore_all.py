# using pol comments exploration code w/ all comments

import pandas as pd
import json
import ijson
import matplotlib
import matplotlib.pyplot as plt

comments_dict = {}
# load comments json, save as dictionary
with open("data/comments.json", "r") as f:
    for key, value in ijson.kvitems(f, ""):
        comments_dict[key] = value

# convert dictionary to df
comments_df = pd.DataFrame.from_dict(comments_dict, orient="index")
comments_df.index.name = "comment_id"

print("comments df:")
print(comments_df.head())
print(comments_df.info())

# save new version
comments_df.to_csv("data/comments-clean.csv")

# load train csv
train_raw = pd.read_csv("data/train-balanced.csv", header=None, names=["raw"])

# parse each row into separate columns, skip rows w/ different format
# currently parentID|childID childID|label other_label
# parsing code from chat
def parse_row(row):
    parts = row.split()

    # check format looks right
    if len(parts) != 3:
        # malformed row
        return pd.Series({
            "context_id": None,
            "comment_id": None,
            "label": None,
            "other_label": None
        })
    if "|" not in parts[0] or "|" not in parts[1]:
        return pd.Series({
            "context_id": None,
            "comment_id": None,
            "label": None,
            "other_label": None
        })

    # split context|comment
    context_id, _ = parts[0].split("|")

    # split comment|label
    comment_id, label = parts[1].split("|")

    # other label
    try:
        other_label = int(parts[2])
    except ValueError:
        other_label = None

    return pd.Series({
        "context_id": context_id,
        "comment_id": comment_id,
        "label": int(label),
        "other_label": other_label
    })


# apply parser
train_df = train_raw["raw"].apply(parse_row)

# drop bad rows
train_df = train_df.dropna(subset=["comment_id"])

print("train df:")
print(train_df.head())
print(train_df.info())

# merge data sets
merged_df = comments_df.merge(
    train_df, 
    left_index=True, 
    right_on="comment_id", 
    how="inner"
)

print("merged df shape:", merged_df.shape)
print(merged_df.head())

# general info
print("general information:")
print("total comments:", len(merged_df))
print("number of authors:", merged_df['author'].nunique())
print("number of subreddits:", merged_df['subreddit'].nunique())
print("sarcasm distribution:")
print(merged_df['label'].value_counts())

# group by author
author_stats = merged_df.groupby('author').agg(
    total_comments=('comment_id', 'count'),
    sarcasm_count=('label', 'sum')
)
author_stats['sarcasm_percent'] = author_stats['sarcasm_count'] * 100 / author_stats['total_comments']

print("info on 10 authors with most comments:")
print(author_stats.sort_values('total_comments', ascending=False).head(10))

# visualizations

# get top 10 authors by total comments
top_authors = author_stats.sort_values('total_comments', ascending=False).head(10)

# bar plot of total comments
plt.figure(figsize=(12,6))
plt.bar(top_authors.index, top_authors['total_comments'], color='blue')
plt.xticks(rotation=45, ha='right')
plt.title("Top 10 Authors with Most Comments")
plt.ylabel("Total Comments")
plt.xlabel("Author")
plt.tight_layout()
plt.show()

# bar plot of sarcasm ratio
plt.figure(figsize=(12,6))
plt.bar(top_authors.index, top_authors['sarcasm_percent'], color='blue')
plt.xticks(rotation=45, ha='right')
plt.title("Sarcasm Percent for Top 10 Authors")
plt.ylabel("Sarcasm Percent")
plt.xlabel("Author")
plt.tight_layout()
plt.show()