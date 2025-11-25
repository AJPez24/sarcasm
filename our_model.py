import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# load embeddings
data = np.load("./data/train_embeddings_pooler.npz")

X = data["embeddings"]      # shape (N, 768)
y = data["labels"]          # shape (N,)

print("Embeddings:", X.shape)
print("Labels:", y.shape)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# model structure
model = Sequential([
    Dense(512, activation="relu", input_shape=(768,)),
    Dropout(0.1),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(128, activation="relu"),
    Dropout(0.1),
    Dense(1, activation="sigmoid")  # binary output
])

# mocel compilation
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# fitting the model 
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=30,
    validation_split=0.1,
    verbose=1
)

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.show()
