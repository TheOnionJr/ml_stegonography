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
import csv

keras.backend.clear_session()

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

AUTOTUNE = tf.data.experimental.AUTOTUNE

MODEL_PATH = ""
CSV_PATH = ""
TR_DIR = ""
IMG_SIZE = (1024, 1024)
BATCH_SIZE = 12
EPOCHS = 10
filepath = "\\tmp\\checkpoint"
func = 'relu'

initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.90,
    staircase=True)

def build_model():
    model = keras.models.Sequential()
    model.add(Conv2D(32,(7,7)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,(5,5),activation=func))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,(5,5),activation=func))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,(5,5),activation=func))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,(3,3),activation=func))
    model.add(Flatten())
    model.add(Dense(288))
    model.add(Dense(256, activation=func))
    model.add(Dense(256, activation=func))
    model.add(Dense(256, activation=func))
    model.add(Dense(256, activation=func))
    model.add(Dense(256, activation=func))
    model.add(Dense(256, activation=func))
    model.add(Dense(256, activation=func))
    model.add(Dense(256, activation=func))
    model.add(Dense(128, activation=func))
    model.add(Dense(128, activation=func))
    model.add(Dense(128, activation=func))
    model.add(Dense(128, activation=func))
    model.add(Dense(128, activation=func))
    model.add(Dense(128, activation=func))
    model.add(Dense(64, activation=func))
    model.add(Dense(32, activation=func))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule,beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
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
    validation_split=0.02,
    subset="validation",
)

#Performance:
tr_dataset = tr_dataset.prefetch(buffer_size=AUTOTUNE)
#val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
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

print(history.history)
model.save(MODEL_PATH)
with open(CSV_PATH, 'a+', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(EPOCHS):
        writer.writerow([ history.history["loss"][i], history.history["accuracy"][i]])
model.summary()
