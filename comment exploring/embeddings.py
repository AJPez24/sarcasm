from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")


# texts = [comments[resp_id]["text"] for resp_id in response_ids]  # list of strings
text = "Replace me by any text you'd like."

encoded_input = tokenizer(text, return_tensors='pt')

output = model(**encoded_input)

embeddings = output.last_hidden_state

print(embeddings.shape)