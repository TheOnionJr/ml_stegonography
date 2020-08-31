import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.models import Model
import os

#from tensorflow.keras.preprocessing.image import image_dataset_from_directory

TR_DIR = "E:\\ds-2"
IMG_SIZE = (1024, 1024)
BATCH_SIZE = 16
EPOCHS = 250

tr_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TR_DIR,
    seed=6600,
    labels="inferred",
    shuffle=True,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size = IMG_SIZE,
    validation_split=0.2,
    subset="validation",
)

input_shape=(BATCH_SIZE,1024,1024,3)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(BATCH_SIZE,1024,1024))
model.add(tf.keras.layers.Dense(2))


model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

tr_dataset = tr_dataset.prefetch(buffer_size=64)


model.fit(
    tr_dataset, epochs=EPOCHS
)
