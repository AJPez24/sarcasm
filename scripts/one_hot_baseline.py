
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#load train and test csvs
train_df = pd.read_csv("./data/responses_flat_train.csv")
test_df = pd.read_csv("./data/responses_flat_test.csv")

#exract text responses and labels from train
train_responses = train_df["response_text"].astype(str).tolist()
train_labels = train_df["label"].tolist()

#exract text responses and labels from test
test_responses = test_df["response_text"].astype(str).tolist()
test_labels = test_df["label"].tolist()

#tokenize training text
vocab_size = 5000  #chose 5000 for 5000 most frequent words
train_tokenizer = Tokenizer(num_words=vocab_size)  #create tokenizer for training data
train_tokenizer.fit_on_texts(train_responses) #learn word frequencies
x_train = train_tokenizer.texts_to_matrix(train_responses, mode="binary")  #one-hot bag-of-words, each text is 5000-dim vector
y_train = np.array(train_labels)

#tokenize test text
test_tokenizer = Tokenizer(num_words=vocab_size) #create tokenizer for test data
test_tokenizer.fit_on_texts(test_responses) #learn word frequencies
x_test = test_tokenizer.texts_to_matrix(test_responses, mode="binary")  #one-hot bag-of-words
y_test = np.array(test_labels)

print("One-hot X shape:", x_train.shape)
print("One-hot Y shape:", y_train.shape)

#four dense layers with dropout to prevent overfitting
model = Sequential([
    Dense(512, activation="relu", input_shape=(vocab_size,)),
    Dropout(0.3),

    Dense(256, activation="relu"),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(1, activation="sigmoid")
])

#prevents model from becoming too confident, binary cross-entropy to measure difference
smoothing_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)

#adamw used in final model and baseline for comparison
adamw = tf.keras.optimizers.AdamW(
            learning_rate=3e-4,
            weight_decay=3e-4 
        )

#compile full model and report accuracy
model.compile(
    loss=smoothing_loss,
    optimizer=adamw,
    metrics=["accuracy"]
)

#callbacks to prevent overfitting
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau( #lowers learning rate when validation loss stops improving
        monitor="val_loss",
        factor=0.5,  #cuts learning rate in half when triggered
        patience=2,  #waits 2 epochs of no improvement before reducing learning rate
        min_lr=1e-6,  #won't reduce lr below this number
        verbose=1 
    ),
    tf.keras.callbacks.EarlyStopping( #early stopping at best epoch
        monitor="val_loss", 
        patience=10, #waits 10 epochs of no improvement before early stopping
        restore_best_weights=True, #restore to best epoch
        verbose=1
    )
]

model.summary()

# fitting the model 
history = model.fit(
    x_train,
    y_train,
    batch_size=16,
    epochs=30,          #max epoch (let callbacks stop early)
    validation_split=0.1,
    callbacks=callbacks, #implement callbacks
    verbose=1
)

#BASELINE MODEL PLOTS
import seaborn as sns
from sklearn.metrics import confusion_matrix

#probability histogram
y_prob = model.predict(X_test).ravel() #.ravel() flattens multidimensional data
plt.hist(y_prob, bins=10, color='navy')
plt.title("Baseline Model Predicted Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
sns.despine()
plt.show()

#confusion matrix
y_pred = (model.predict(X_test) > 0.5).astype(int) #threshold at 0.5 as standard
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

# plot loss
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.show()