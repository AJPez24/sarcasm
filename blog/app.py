# Adapted code from ChatGPT for flask implementation
from flask import Flask, render_template, request, jsonify

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tensorflow.keras.models import load_model

# Loading BERT and classifier models
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

clf_model = load_model("./models/final_sarcasm_model.h5")

app = Flask(__name__)


# Converts input text into BERT embedding
def preprocess_text(text: str) -> np.ndarray:
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        output = bert_model(**encoding)
        token_embeddings = output.last_hidden_state
        # Using mean pooling
        current_embedding = token_embeddings.mean(dim=1)
        current_embedding = current_embedding.squeeze().numpy()

    # Output is a 768-dimension vector
    return current_embedding.reshape(1, -1)

# Takes a sentence as input and returns the sarcasm label and probability
def predict_sentence(text: str):
    # Generates the BERT embedding
    x = preprocess_text(text)
    # Runs the embedding through the classifier model
    prob = float(clf_model.predict(x, verbose=0)[0][0])
    # Classify with threshold of 0.4
    label = "sarcastic" if prob >= 0.4 else "not sarcastic"
    return label, prob


# Render and run the actual HTML website
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Empty text"}), 400

    label, prob = predict_sentence(text)
    return jsonify({"label": label, "probability": prob})


if __name__ == "__main__":
    app.run(debug=True)
