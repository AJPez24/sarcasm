# Model file 

# imports 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

# load embeddings attained from bert 
train_data = np.load("./data/train_embeddings_mean.npz")
test_data = np.load("./data/test_embeddings_mean.npz")

x_train = train_data["embeddings"]      # shape (N, 768)
y_train = train_data["labels"]          # shape (N,)

x_test = test_data["embeddings"]      # shape (N, 768)
y_test = test_data["labels"]          # shape (N,)

print("Embeddings:", x_train.shape)
print("Labels:", y_train.shape)

# model structure
model = Sequential([
    Dense(512, activation="relu", input_shape=(768,)),
    Dropout(0.3), # dropout to avoid overfitting 

    Dense(256, activation="relu"),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(1, activation="sigmoid")
])


# smoothing loss allows our model to avoid extremities in probability 
smoothing_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)

# adamW optimizer
adamw = tf.keras.optimizers.AdamW(
            learning_rate=3e-4,
            weight_decay=3e-4 
        )

# model compilation 
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
    epochs=20,          # let callbacks stop early
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

# evaluate on held-out test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# calibration curve
# calibration curves:
# numerical predicaiton and targets _> check model accuracy scores agaisnt diff decision thresholds
# -> for diff sensitivities of the output 
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# predicted probabilities on test set
y_prob = model.predict(x_test, verbose=0).ravel()  # shape (N_test,)

# compute calibration curve
prob_true, prob_pred = calibration_curve(
    y_test, y_prob,
    n_bins=10,        # number of bins
    strategy='quantile'  # each bin has ~same # of samples
)

# plot reliability (calibration curve) diagram
plt.figure()
plt.plot(prob_pred, prob_true, marker="o", linewidth=1, label="Model")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
plt.xlabel("Predicted probability")
plt.ylabel("Observed fraction of positives")
plt.title("Calibration curve (reliability diagram)")
plt.legend()
plt.grid(True)
plt.show()

# threshold search 
from sklearn.metrics import f1_score

best_thr = 0.5
best_f1 = -1

for thr in np.linspace(0.1, 0.9, 17):  # 0.1 → 0.9 in steps of 0.05
    preds = (y_prob >= thr).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print("Best threshold:", best_thr)
print("F1 at best threshold:", best_f1)

# save model to be later loaded into app.py
model.save("sarcasm_file_withcalib.h5")