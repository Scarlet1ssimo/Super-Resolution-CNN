# %%
import tensorflow as tf
from tensorflow import keras
import numpy as npy
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import SR_import as imp


def model():
    SRCNN = keras.Sequential()
    SRCNN.add(keras.layers.Convolution2D(
        filters=128,
        kernel_size=9,
        strides=1,
        padding='valid',
        activation='relu',
        input_shape=(None, None, 1)  # 大小不限
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


def predict():
    srcnn_model = model()
    srcnn_model.load_weights(imp.NMD)
    IMG_NAME = "Test/Set5/baby_GT.bmp"
    INPUT_NAME = "input2.jpg"
    OUTPUT_NAME = "pre2.jpg"

    import cv2
    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    im1 = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    shape = img.shape
    Y_img = cv2.resize(img[:, :, 0], (shape[1] // imp.scale,
                                      shape[0] // imp.scale), cv2.INTER_CUBIC)
    Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    img[:, :, 0] = Y_img
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    im2 = img
    cv2.imwrite(INPUT_NAME, img)

    Y = npy.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.
    pre = srcnn_model.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(npy.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img[imp.margin: -imp.margin, imp.margin: -imp.margin, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    im3 = img
    cv2.imwrite(OUTPUT_NAME, img)

    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(im1)
    plt.xlabel("qwq")
    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(im2)
    plt.xlabel("bicubic:{}".format(cv2.PSNR(im1, im2)))
    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(im3)
    plt.xlabel("SRCNN:{}".format(cv2.PSNR(im1, im3)))
    plt.show()


if __name__ == "__main__":
    predict()
