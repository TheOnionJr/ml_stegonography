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
EPOCHS = 5
filepath = "\\tmp\\checkpoint"
func = 'relu'

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

model = load_model(MODEL_PATH)
while True:
    model.fit(
        tr_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        use_multiprocessing=True,
        workers=8,
    )
    model.summary()
    model.save(MODEL_PATH)
