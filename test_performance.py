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
import time

holdout_dir = ""
MODEL_PATH = ""
AUTOTUNE = tf.data.experimental.AUTOTUNE

model = keras.models.load_model(MODEL_PATH)

t = time.time()

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "",
    seed=6600,
    labels="inferred",
    shuffle=False,
    color_mode="rgb",
    batch_size=10,
    image_size = (1024,1024),
)
dataset = dataset.prefetch(buffer_size=AUTOTUNE)

predictions = model.predict(dataset, verbose=0)

print(predictions)
print(f'Elapsed time: {str(time.time()-t)}')
