import pandas as pd
import numpy as np
from transformers import TFBertModel, BertTokenizer
import tensorflow as tf

BATCH_SIZE = 32

CSV_PATH = "data/responses_flat.csv"

df = pd.read_csv(CSV_PATH)
texts = df["response_text"].tolist()
labels = df ["label"].astype(int).values


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = TFBertModel.from_pretrained("bert-base-uncased")

#TF dataset loader 
def batch_tokenize(text_batch):
    return tokenizer(text_batch, padding = True, truncation = True,
                     max_length=128, return_tensors='tf')


# generate embeddings tbd



# texts = [comments[resp_id]["text"] for resp_id in response_ids]  # list of strings
text = "Replace me by any text you'd like."

encoded_input = tokenizer(text, return_tensors='pt')

output = model(**encoded_input)

embeddings = output.last_hidden_state

print("Embeddings shape:", embeddings.shape)
print("Labels shape:", labels_arr.shape)