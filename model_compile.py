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

MODEL_PATH = "E:\\models"
TR_DIR = "E:\\ds-5"
IMG_SIZE = (1024, 1024)
BATCH_SIZE = 8
EPOCHS = 10
filepath = "\\tmp\\checkpoint"
func = 'relu'

def build_model():
    model = keras.models.Sequential()
    model.add(Conv2D(32,(7,7)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,(5,5),activation=func))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(256, activation=func))
    model.add(Dense(128, activation=func))
    model.add(Dense(64, activation=func))
    model.add(Dense(32, activation=func))
    model.add(Dense(2, activation=func))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-07),
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
        filepath="E:/tmp/t_save.h5",
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

model = model.fit(
    tr_dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    use_multiprocessing=True,
    workers=8,
    callbacks=callbacks,
)

model.summary()

model.save(MODEL_PATH)
