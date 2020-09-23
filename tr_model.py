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

LOG_DIR = f"{int(time.time())}"

TR_DIR = "E:\\ds-2\\train"
IMG_SIZE = (1024, 1024)
BATCH_SIZE = 24
EPOCHS = 500
AUTOTUNE = tf.data.experimental.AUTOTUNE

tr_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TR_DIR,
    seed=6600,
    labels="inferred",
    shuffle=True,
    color_mode="rgb",
    #batch_size=BATCH_SIZE,
    image_size = IMG_SIZE,
    validation_split=0.2,
    subset="validation",
)
tr_dataset = tr_dataset.prefetch(buffer_size=32)


def build_model(hp):
    model = keras.models.Sequential()
    model.add(Dense(hp.Int("input_nodes",8,16,8)))
    #model.add(Conv2D(18,hp.Int("input_kernel_size",1,7,2),activation="relu"))
    #model.add(MaxPooling2D(pool_size=(7,7)))

    #for i in range(hp.Int("n_layers",0,4,1)):
    #    model.add(Conv2D(hp.Int(f"conv_{i}_units",2,10,1),hp.Int(f"conv_{i}_kernel",1,7,2),activation="relu"))
    for i in range(hp.Int("n_layers",0,4,1)):
        model.add(Dense(hp.Int(f"dense_nodes{i}",2,32,2)))

    model.add(Flatten())
    model.add(Dense(2))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model



tuner = RandomSearch(
    build_model,
    objective = "accuracy",
    max_trials = 32,
    executions_per_trial = 3,
    directory = LOG_DIR
)
tuner.search(
    tr_dataset,
    epochs=8,
    batch_size=BATCH_SIZE
)

print(tuner.get_best_hyperparameters()[0].values)
print(tuner.results_summary())
