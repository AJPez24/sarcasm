from flask import Flask, render_template, request, jsonify

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tensorflow.keras.models import load_model

# -------------------------
# LOAD BERT + CLASSIFIER
# -------------------------

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

clf_model = load_model("sarcasm_model.h5")

app = Flask(__name__)


# -------------------------
# TEXT → BERT MEAN EMBEDDING
# (matches your training script)
# -------------------------

def preprocess_text(text: str) -> np.ndarray:
    # same as training: truncation=True, no padding/max_length specified
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        output = bert_model(**encoding)
        token_embeddings = output.last_hidden_state          # (1, seq_len, 768)
        current_embedding = token_embeddings.mean(dim=1)     # (1, 768)
        current_embedding = current_embedding.squeeze().numpy()  # (768,)

    return current_embedding.reshape(1, -1)                  # (1, 768)


def predict_sentence(text: str):
    x = preprocess_text(text)
    prob = float(clf_model.predict(x, verbose=0)[0][0])      # sigmoid output
    label = "sarcastic" if prob >= 0.35 else "not sarcastic"
    return label, prob


# -------------------------
# ROUTES
# -------------------------

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
