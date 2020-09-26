import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Flatten, Dense
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from keras.models import Model
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import time
import os
import matplotlib.pyplot as plt
import numpy

keras.backend.clear_session()

AUTOTUNE = tf.data.experimental.AUTOTUNE

TR_DIR = "E:\\ds-1\\train"
IMG_SIZE = (1024, 1024)
BATCH_SIZE = 10
EPOCHS = 60
filepath = "\\tmp\\checkpoint"
func = 'relu'

def build_model():
    model = keras.models.Sequential()

    model.add(Conv2D(26,(3,3),activation=func))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(26,(3,3),activation=func))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation=func))
    model.add(Dense(125, activation=func))
    model.add(Dense(125, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(2))

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model

def show_accuracy():
    plt.plot((history.history['accuracy'])*10)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="E:/tmp/save_{epoch}.h5",
    )
]

#DATA Ingestion
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

#Performance:
tr_dataset = tr_dataset.prefetch(buffer_size=AUTOTUNE)
#############

model = build_model()

history = model.fit(
    tr_dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    use_multiprocessing=True,
    workers=8,
    callbacks=callbacks,
)
model.summary()
show_accuracy()
