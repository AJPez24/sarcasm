#INITIAL MODEL

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data = np.load("./data/bert_embeddings.npz")

X = data["embeddings"]      # shape (N, 768)
y = data["labels"]          # shape (N,)

print("Embeddings:", X.shape)
print("Labels:", y.shape)

# train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
# model structure 3 dense layers
model = Sequential([
    Dense(512, activation="relu", input_shape=(768,)),
    Dropout(0.2),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")  # binary output
])


# model compilation
model.compile(
    loss="binary_crossentropy", #for binary predictions
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

#PLOTS FOR INITIAL MODEL

import seaborn as sns

#setting theme for plots
sns.set_theme(style = "ticks")

#confusion matrix
y_pred = (model.predict(X_test) > 0.5).astype(int) #threshold 0.5, calculating y_pred from test data
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Reds", 
            xticklabels=['Non-Sarcastic', 'Sarcastic'],
            yticklabels=['Non-Sarcastic', 'Sarcastic'])
plt.title("Initial Model Confusion Matrix - 30 Epochs ")
plt.show()

#probability histogram
y_prob = model.predict(X_test).ravel() # .ravel() flattens multidimensional data
plt.hist(y_prob, bins=30, color='darkred')
plt.title("Initial Model Predicted Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
sns.despine()
plt.show()


#loss plot
plt.plot(history.history["loss"], label="Train Loss", color='deepskyblue')
plt.plot(history.history["val_loss"], label="Val Loss", color = 'darkred')
plt.title("Initial Model Loss")
plt.xlabel("Number of Epochs")
sns.despine()
plt.ylabel("Loss")
plt.legend()
plt.show()