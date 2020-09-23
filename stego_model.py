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
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

#from tensorflow.keras.preprocessing.image import image_dataset_from_directory

keras.backend.clear_session()

TR_DIR = "E:\\ds-3"
IMG_SIZE = (1024, 1024)
BATCH_SIZE = 2
EPOCHS = 1400
filepath = "\\tmp\\checkpoint"

def build_model():
    model = keras.models.Sequential()
    model.add(Conv2D(14,1,activation="relu"))
    model.add(Dense(6))
    model.add(Flatten())
    model.add(Dense(2))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="E:/tmp/save_{epoch}.h5",
    )
]


tr_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TR_DIR,
    seed=6600,
    labels="inferred",
    shuffle=True,
    color_mode="rgb",
    #batch_size=BATCH_SIZE,
    image_size = IMG_SIZE
)

model = build_model()

tr_dataset = tr_dataset.prefetch(buffer_size=528)


model.fit(
    tr_dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    use_multiprocessing=True,
    workers=8,
    callbacks=callbacks
)
model.summary()
