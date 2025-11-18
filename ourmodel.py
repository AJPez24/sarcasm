import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

#X_train_emb doesn't exit yet - these are the embeddings from BERT
# need y train also - which is the train-balanced labels
emb_dim = X_train_emb.shape[1]    # dimension 768 from BERT

# model structure
model = Sequential([
    Dense(512, activation="relu", input_shape=(emb_dim,)),
    Dropout(0.2),
    Dense(128, activation="relu"),
    Dropout(0.2),
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
""" history = model.fit(
    X_train_emb,
    y_train,
    batch_size=32,
    epochs=5,
    validation_split=0.1,
    verbose=1
)

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.show()
 """