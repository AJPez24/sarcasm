import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/responses_flat_more_columns.csv")

# plot - how many sarcastic vs non-sarcastic
# equal because balanced dataset
label_counts = df['label'].value_counts().sort_index()

plt.figure(figsize=(6,4))
plt.bar(label_counts.index.astype(str), label_counts.values)
plt.xticks(ticks=[0, 1], labels=["Not Sarcastic", "Sarcastic"])
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Sarcasm Labels Count")
plt.show()


# top authors by % of 1s
author_stats = df.groupby('author')['label'].agg(['mean', 'count'])
# filter authors with at least 5 comments to avoid noise
author_stats = author_stats[author_stats['count'] >= 5]
top_authors = author_stats.sort_values('mean', ascending=False).head(10)

plt.figure(figsize=(10,5))
plt.bar(top_authors.index, top_authors['mean'])
plt.xticks(rotation=45, ha='right')
plt.title("Top 10 Authors by Sarcasm Percent")
plt.xlabel("Author")
plt.ylabel("Percentage of Comments that are Sarcastic")
plt.ylim(0,1)
plt.tight_layout()
plt.show()

# top subreddits by % of 1s
subreddit_stats = df.groupby('subreddit')['label'].agg(['mean', 'count'])
# filter subreddits with at least 5 comments
subreddit_stats = subreddit_stats[subreddit_stats['count'] >= 5]
top_subreddits = subreddit_stats.sort_values('mean', ascending=False).head(10)

plt.figure(figsize=(10,5))
plt.bar(top_subreddits.index, top_subreddits['mean'])
plt.xticks(rotation=45, ha='right')
plt.title("Top 10 Subreddits by Sarcasm Percent")
plt.xlabel("Subreddit")
plt.ylabel("Percentage of Comments that are Sarcastic")
plt.ylim(0,1)
plt.tight_layout()
plt.show()



# filter only rows with label == 1
df_ones = df[df['label'] == 1]

# top Authors with most 1s
top_authors = df_ones['author'].value_counts().head(10)
plt.figure(figsize=(10,5))
plt.bar(top_authors.index, top_authors.values)
plt.xticks(rotation=45, ha='right')
plt.title("Top 10 Authors with Most Sarcastic Comments")
plt.xlabel("Author")
plt.ylabel("Sarcastic Comments")
plt.tight_layout()
plt.show()

# top Subreddits with most 1s
top_subreddits = df_ones['subreddit'].value_counts().head(10)
plt.figure(figsize=(10,5))
plt.bar(top_subreddits.index, top_subreddits.values)
plt.xticks(rotation=45, ha='right')
plt.title("Top 10 Subreddits with Most Sarcastic Comments")
plt.xlabel("Subreddit")
plt.ylabel("Sarcastic Comments")
plt.tight_layout()
plt.show()
