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

AUTOTUNE = tf.data.experimental.AUTOTUNE

MODEL_PATH = "model"
CSV_PATH = "data\\history1.csv"
TR_DIR = "E:\\ds-5"
VAL_DIR = "E:\\ds-5\\validate"
IMG_SIZE = (1024, 1024)
BATCH_SIZE = 16
EPOCHS = 20
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

model = keras.models.load_model(MODEL_PATH)
while True:
    history = model.fit(
        tr_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        use_multiprocessing=True,
        workers=8,
    )
    model.save(MODEL_PATH)
    with open(CSV_PATH, 'a+', newline='\n') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(EPOCHS):
            writer.writerow([ history.history["loss"][i], history.history["accuracy"][i]])
    model.summary()
