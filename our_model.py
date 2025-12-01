import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load embeddings
data = np.load("./data/train_embeddings_mean.npz")

x = data["embeddings"]      # shape (N, 768)
y = data["labels"]          # shape (N,)

print("Embeddings:", x.shape)
print("Labels:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42, stratify=y
)

model = Sequential([
    Dense(512, activation="relu", input_shape=(768,)),
    Dropout(0.3),

    Dense(256, activation="relu"),
    Dropout(0.2),

    Dense(64, activation="relu"),
    Dropout(0.1),

    Dense(1, activation="sigmoid")
])



adamw = tf.keras.optimizers.AdamW(
            learning_rate=1e-3,
            weight_decay=1e-4 
        )

model.compile(
    loss="binary_crossentropy",
    optimizer=adamw,
    metrics=["accuracy"]
)


callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
]

model.summary()

# fitting the model 
history = model.fit(
    X_train,
    y_train,
    batch_size=8,
    epochs=10,          # let callbacks stop early
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# plot loss
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.show()

# optional: evaluate on held-out test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)