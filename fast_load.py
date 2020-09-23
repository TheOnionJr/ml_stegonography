#experimental script for loading data faster
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Flatten, Dense
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from keras.models import Model
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import numpy as np
import time
import os


def build_model(hp):
    model = keras.models.Sequential()
    model.add(Conv2D(hp.Int("input_units",2,10,1),hp.Int("input_kernel",1,7,2),activation="relu"))
    model.add(MaxPooling2D(pool_size=(7,7)))

    for i in range(hp.Int("n_conv_layers",1,4,1)):
        model.add(Conv2D(hp.Int(f"conv_{i}_units",2,10,1),hp.Int(f"conv_{i}_kernel",1,7,2),activation="relu"))
    model.add(Flatten())
    for i in range(hp.Int("n_dense_layers",1,4,1)):
        model.add(Dense(hp.Int(f"dense_nodes{i}",2,32,2)))


    model.add(Dense(2)) #Number of categories to identify

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model

def load_data(folder):
    #Inspierd by the article below
    #https://medium.com/swlh/dump-keras-imagedatagenerator-start-using-tensorflow-tf-data-part-1-a30330bdbca9
    #Modified to make more sence and work for windows
    def ingest_image(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        class_names = np.array(os.listdir(DATASET + '\\train'))
        label = parts[-2] == class_names
        print(label)
        print(file_path)
        img = tf.io.read_file(file_path)
        print(img)
        img = tf.image.decode_jpeg(img, channels=3)
        print(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        print(img)
        img = tf.image.resize(img, [1024, 1024])
        print(img)
        return img#, label

    def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.repeat(10000)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    list_ds = tf.data.Dataset.list_files(str(folder+'\\*\\*'))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    labeled_ds = list_ds.map(
        ingest_image, num_parallel_calls=AUTOTUNE)
    print(labeled_ds)
    dataset = prepare_for_training(
        labeled_ds, cache='data.tfcache')

    return dataset


LOG_DIR = f"{int(time.time())}"

DATASET = "E:\\ds-2"

TR_DIR = DATASET + "\\train"
VL_DIR = DATASET + "\\validate"

FOLDERS = ["train","validate"]
CATEGORIES = ["cover","stego"]

IMG_SIZE = (1024, 1024)
BATCH_SIZE = 32
EPOCHS = 500

tr_set = load_data(TR_DIR)
print(tr_set)



tuner = RandomSearch(
    build_model,
    objective = "accuracy",
    max_trials = 120,
    executions_per_trial = 1,
    directory = LOG_DIR
)
tuner.search(
    tr_set,
    epochs=3,
    batch_size=BATCH_SIZE
)

print(tuner.get_best_hyperparameters()[0].values)
print(tuner.results_summary())
