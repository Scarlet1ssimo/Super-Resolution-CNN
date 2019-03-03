import tensorflow as tf
from tensorflow import keras
import numpy as npy
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import SR_import as imp
import SR_predict as pred


def qwq(path, name):
    srcnn_model = pred.model()
    srcnn_model.load_weights(imp.NMD)
    IMG_NAME = path
    OUTPUT_NAME = name

    import cv2
    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    shape = img.shape
    img = cv2.resize(
        img, (shape[1] * imp.scale, shape[0] * imp.scale), cv2.INTER_CUBIC)
    Y_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y_img = Y_img[:, :, 0]
    #img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

    Y = npy.zeros((1, img.shape[0],
                   img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.
    pre = srcnn_model.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(npy.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img[imp.margin: -imp.margin, imp.margin: -imp.margin, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(OUTPUT_NAME, img)


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        qwq(sys.argv[1], sys.argv[2])
    else:
        qwq("Original.jpg", "qwq3.jpg")
