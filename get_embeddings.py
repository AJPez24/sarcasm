import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

model.eval()

df = pd.read_csv("./data/responses_flat.csv")

ids = df["response_id"].tolist()
responses = df["response_text"].astype(str).tolist()
labels = df["label"].tolist()

embeddings = []

#Get the embedding for each response and add it to the embedding list
for response in tqdm(responses, desc="Embedding responses"):
    encoding = tokenizer(response, return_tensors='pt', truncation=True)

    with torch.no_grad():
        output = model(**encoding)

    current_embedding = output.pooler_output[0].numpy() #should also try "last_hidden_state"
    embeddings.append(current_embedding)

#convert to numpy arrays
embeddings = np.vstack(embeddings)
labels = np.array(labels)
ids = np.array(ids)

print("Embeddings shape:", embeddings.shape)
print("Labels shape:", labels.shape)

#save in a new npz file
np.savez("./data/bert_embeddings.npz", embeddings=embeddings, labels=labels, ids=ids)

print("Saved to bert_embeddings.npz")