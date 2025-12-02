#creating csv file for training data stripped of punctuation
import pandas as pd

df = pd.read_csv("./data/responses_flat_train.csv")

df["response_text"] = (
    df["response_text"]
    .str.lower()
    .str.replace(r"[^\w\s',]", "", regex=True)
)

df.to_csv("./data/strip_responses_train.csv", index=False)


#creating csv file for testing data stripped of punctuation
df = pd.read_csv("./data/responses_flat_test.csv")

df["response_text"] = (
    df["response_text"]
    .str.lower()
    .str.replace(r"[^\w\s',]", "", regex=True)
)

df.to_csv("./data/strip_responses_test.csv", index=False)
