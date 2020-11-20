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

MODEL_PATH = ""
CSV_PATH = ""
TR_DIR = ""
VAL_DIR = ""
IMG_SIZE = (1024, 1024)
BATCH_SIZE = 10
EPOCHS = 75
filepath = "\\tmp\\checkpoint"
holdout_dir = ""
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
    validation_split=0.05,
    subset="validation",
)

#Performance:
tr_dataset = tr_dataset.prefetch(buffer_size=AUTOTUNE)
#############

#Load model
model = keras.models.load_model(MODEL_PATH)

#Repeatedly train until cancelled by user with ctrl + c
while True:
    history = model.fit(
        tr_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        use_multiprocessing=True,
        workers=8,
    )

    model.save(MODEL_PATH)

    #Write history data to csv file
    with open(CSV_PATH, 'a+', newline='\n') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(EPOCHS):
            writer.writerow([ history.history["loss"][i], history.history["accuracy"][i]])

    #PREDICTIONS:

    #load holdout dataset
    hold_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        holdout_dir,
        seed=6600,
        labels="inferred",
        shuffle=False,
        color_mode="rgb",
        batch_size=10,
        image_size = (1024,1024),
    )

    #Run predictions and evaluations
    predictions = model.predict(hold_dataset, verbose=0)
    eval = model.evaluate(hold_dataset, verbose=0)

    #Print results
    print(f'Test loss: {eval[0]} / Test accuracy: {eval[1]}')
    print("predictions shape:", predictions.shape)
    print(predictions)
