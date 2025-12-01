import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/responses_flat_train.csv")

responses = df["response_text"].astype(str).tolist()
labels = df["label"].tolist()

# Tokenize text
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(responses)
x = tokenizer.texts_to_matrix(responses, mode="binary")  # one-hot bag-of-words

y = np.array(labels)

print("One-hot X shape:", x.shape)
print("One-hot Y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42, stratify=y
)

model = Sequential([
    Dense(512, activation="relu", input_shape=(vocab_size,)),
    # Dropout(0.1),
    Dense(128, activation="relu"),
    # Dropout(0.2),
    Dense(128, activation="relu"),
    # Dropout(0.1),
    Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=1,
    validation_split=0.1,
    verbose=1
)

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.show()
