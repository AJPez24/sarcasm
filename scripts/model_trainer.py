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


x_train = train_data["embeddings"]      #shape (n, 768)
y_train = train_data["labels"]          #shape (n,)

x_test = test_data["embeddings"]      #shape (n, 768)
y_test = test_data["labels"]          #shape (n,)

print("Embeddings:", x_train.shape)
print("Labels:", y_train.shape)

#four dense layers
model = Sequential([
    Dense(512, activation="relu", input_shape=(768,)),
    Dropout(0.3),

    Dense(256, activation="relu"),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(1, activation="sigmoid") #binary output and single classification
])


#softens target labels to prevent overfitting
smoothing_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)

#using adamw optimizer because it separates weight gradients and regularization
adamw = tf.keras.optimizers.AdamW(
            learning_rate=3e-4,
            weight_decay=3e-4 
        )

model.compile(
    loss=smoothing_loss,
    optimizer=adamw,
    metrics=["accuracy"] #reports accuracy for training and validation
)

#training callbacks control learning behavior
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(  #lowers learning rate when validation loss stops improving
        monitor="val_loss",
        factor=0.5,  #cuts learning rate in half when triggered
        patience=2,  #waits 2 epochs of no improvement before reducing learning rate
        min_lr=1e-6,  #won't reduce lr below this number
        verbose=1 
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,  #waits 10 epochs of no improvement before early stopping
        restore_best_weights=True, #restore to best performing weights
        verbose=1
    )
]

model.summary()

# fitting the model 
history = model.fit(
    x_train,
    y_train,
    batch_size=16,  #number of samples processed before the weights update
    epochs=20,          #max number of epochs (let callbacks stop early)
    validation_split=0.1,
    callbacks=callbacks,  #apply callbacks
    verbose=1
)

# PLOTS FOR FINAL MODEL

from sklearn.metrics import confusion_matrix
import seaborn as sns

#for visuals
sns.set_theme(style = "ticks")

#confusion matrix
y_pred = (model.predict(x_test) > 0.4).astype(int)  #threshold of 0.4 based off of calibration curve
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',  
            xticklabels=['Non-Sarcastic', 'Sarcastic'],
            yticklabels=['Non-Sarcastic', 'Sarcastic'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Final Model Confusion Matrix")
plt.show()


#probability histogram
y_prob = model.predict(x_test).ravel() #.ravel() flattens multidimensional data
plt.hist(y_prob, bins=10, color='darkgreen')
plt.title("Final Model Predicted Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
sns.despine()  #remove spines from graph
plt.show()


#loss plot
plt.plot(history.history["loss"], label="Train Loss", color='darkorange')
plt.plot(history.history["val_loss"], label="Val Loss", color = 'darkgreen')
plt.title("Final Model Loss")
plt.xlabel("Number of Epochs")
sns.despine()  #remove spines from graph
plt.ylabel("Loss")
plt.legend()
plt.show()