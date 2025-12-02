import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

# load embeddings
train_data = np.load("./data/train_embeddings_mean.npz")
test_data = np.load("./data/test_embeddings_mean.npz")

x_train = train_data["embeddings"]      # shape (N, 768)
y_train = train_data["labels"]          # shape (N,)

x_test = test_data["embeddings"]      # shape (N, 768)
y_test = test_data["labels"]          # shape (N,)

print("Embeddings:", x_train.shape)
print("Labels:", y_train.shape)


model = Sequential([
    Dense(512, activation="relu", input_shape=(768,)),
    Dropout(0.3),

    Dense(256, activation="relu"),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(1, activation="sigmoid")
])



smoothing_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)

adamw = tf.keras.optimizers.AdamW(
            learning_rate=3e-4,
            weight_decay=3e-4 
        )

model.compile(
    loss=smoothing_loss,
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
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]

model.summary()

# fitting the model 
history = model.fit(
    x_train,
    y_train,
    batch_size=16,
    epochs=30,          # let callbacks stop early
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
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

model.save("sarcasm_file.h5")