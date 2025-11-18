import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

emb_dim = X_train_emb.shape[1]      # dimension 768 from BERT

model = Sequential([
    Dense(128, activation="relu", input_shape=(emb_dim,)),
    Dropout(0.3),
    Dense(1, activation="sigmoid")  # binary output
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train_emb,
    y_train,
    batch_size=32,
    epochs=5,
    validation_split=0.1,
    verbose=1
)
