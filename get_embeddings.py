import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

model.eval()

df = pd.read_csv("./data/new_responses_flat_train.csv")

response_ids = df["response_id"].tolist()
responses = df["response_text"].astype(str).tolist()

comment_ids = df["main_comment_id"].tolist()
comments = df["main_comment_text"].tolist()

labels = df["label"].tolist()

embeddings = []

#Get the embedding for each response and add it to the embedding list
for comment, response in tqdm(zip(comments, responses), desc="Embedding comment-response pairs", total=len(responses)):
    text = (comment + " " + response)

    encoding = tokenizer(text, return_tensors='pt', truncation=True)

    with torch.no_grad():
        output = model(**encoding)

    #mean pooling
    token_embeddings = output.last_hidden_state
    current_embedding = token_embeddings.mean(dim=1).squeeze().numpy()

    embeddings.append(current_embedding)

#convert to numpy arrays
embeddings = np.vstack(embeddings)
labels = np.array(labels)
ids = np.array(ids)

print("Embeddings shape:", embeddings.shape)
print("Labels shape:", labels.shape)

#save in a new npz file
np.savez("./data/paired_train_embeddings_mean.npz", embeddings=embeddings, labels=labels, ids=ids)

print("Saved to paired_train_embeddings_mean.npz")