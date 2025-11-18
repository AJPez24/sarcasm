# using pol comments exploration code w/ all comments

import pandas as pd
import ijson
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
    # deletes leading & trailing whitespace characters
    row = row.strip()
    # split into the 3 sections
    parts = row.split("|")
    if len(parts) != 3:
        return None

    context_str, response_str, label_str = parts

    # split inside each section
    context_ids = context_str.strip().split()
    response_ids = response_str.strip().split()
    labels = label_str.strip().split()

    # convert label strings to ints
    try:
        labels = list(map(int, labels))
    except ValueError:
        return None

    # check
    if len(response_ids) != len(labels):
        return None

    return pd.Series({
        "context_ids": context_ids,
        "response_ids": response_ids,
        "labels": labels
    })


# apply parser
parsed = train_raw["raw"].apply(parse_row).dropna()

# currently a row w/ a list in each column
# explode into multiple rows to get 1 item per row
expanded = parsed.explode(["response_ids", "labels"])
expanded = expanded.rename(columns={
    "response_ids": "comment_id",
    "labels": "label"
})

print("train df:")
print(expanded.head())
print(expanded.info())

# merge data sets
merged_df = comments_df.merge(
    expanded, 
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