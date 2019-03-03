# %%import
import tensorflow as tf
from tensorflow import keras
import numpy as npy
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import SR_import as imp
import SR_predict as pre
from PIL import Image
# %%


def model():
    SRCNN = keras.Sequential()
    SRCNN.add(keras.layers.Convolution2D(
        filters=128,
        kernel_size=9,
        strides=1,
        padding='valid',
        activation='relu',
        input_shape=(imp.patch_size, imp.patch_size, 1)
    ))
    SRCNN.add(keras.layers.Convolution2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu'
    ))
    SRCNN.add(keras.layers.Convolution2D(
        filters=1,
        kernel_size=5,
        strides=1,
        padding='valid',
        activation='linear'
    ))
    SRCNN.compile(optimizer=keras.optimizers.Adam(lr=0.0003), loss='mean_squared_error',
                  metrics=['mean_squared_error'])
    return SRCNN


def train():
    SRCNN = model()
    print(SRCNN.summary())
    data, label = imp.load(imp.PATH)


    val_data, val_label = imp.load(imp.TEST)
    checkpoint = keras.callbacks.ModelCheckpoint(imp.NMD, monitor='val_loss', verbose=1, save_best_only=True,
                                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    SRCNN.fit(data, label, batch_size=128, validation_data=(val_data, val_label),
              callbacks=callbacks_list, shuffle=True, nb_epoch=20)


if __name__ == "__main__":
    train()
    pre.predict()
