import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
import seaborn as sns


train_df = pd.read_csv("./data/responses_flat_train.csv")
test_df = pd.read_csv("./data/responses_flat_test.csv")

train_responses = train_df["response_text"].astype(str).tolist()
train_labels = train_df["label"].tolist()

test_responses = test_df["response_text"].astype(str).tolist()
test_labels = test_df["label"].tolist()

# Tokenize text
vocab_size = 5000
train_tokenizer = Tokenizer(num_words=vocab_size)
train_tokenizer.fit_on_texts(train_responses)
x_train = train_tokenizer.texts_to_matrix(train_responses, mode="binary")  # one-hot bag-of-words
y_train = np.array(train_labels)

test_tokenizer = Tokenizer(num_words=vocab_size)
test_tokenizer.fit_on_texts(test_responses)
x_test = test_tokenizer.texts_to_matrix(test_responses, mode="binary")  # one-hot bag-of-words
y_test = np.array(test_labels)

print("One-hot X shape:", x_train.shape)
print("One-hot Y shape:", y_train.shape)

model = Sequential([
    Dense(512, activation="relu", input_shape=(vocab_size,)),
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


#probability histogram
y_prob = model.predict(x_test).ravel()
plt.hist(y_prob, bins=10, color='navy')
plt.title("Baseline Model Predicted Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
sns.despine()
plt.show()

#confusion matrix

y_prob = model.predict(x_test)
y_pred = (y_prob > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Non-Sarcastic', 'Sarcastic'],
    yticklabels=['Non-Sarcastic', 'Sarcastic']
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Baseline Model Confusion Matrix")
plt.show()