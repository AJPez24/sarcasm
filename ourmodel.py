
import numpy as np
import matplotlib.pylab as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist

model = Sequential([
    Conv2D(filters=10, kernel_size=(3,3), strides=(1, 1), input_shape=[28,28,1], padding='valid'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(10, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()




