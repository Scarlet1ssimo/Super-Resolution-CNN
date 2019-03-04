# %% import
import tensorflow as tf
from tensorflow import keras

import numpy as npy
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
from PIL import Image


scale = 3
patch_size = 33
label_size = 21
margin = 6
stride = 14
NMD = "SRCNN" + str(scale) + ".h5"
TRAIN = "Train/"
TEST = "Test/Set5/"
# 2,4:22910  3:21910


def load(qwq, nmd=0):
    cnt = 0
    k = 0
    images_paths = glob.glob(qwq + "*.bmp")
    images_paths += glob.glob(qwq + "*.jpg")
    #images_paths += glob.glob(qwq+"*.png")
    for path in images_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        shape = img.shape
        for j in range(stride, shape[1]+stride-patch_size, stride):
            for i in range(stride, shape[0] + stride-patch_size, stride):
                cnt += 1

    data = npy.zeros((cnt, patch_size, patch_size, 1), dtype=npy.double)
    label = npy.zeros(
        (cnt, label_size, label_size, 1), dtype=npy.double)
    cnt = 0

    for path in images_paths:
        k = k+1
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        shape = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = img[:, :, 0]

        reimg = cv2.resize(
            img, (shape[1] // scale, shape[0] // scale), cv2.INTER_CUBIC)
        reimg = cv2.resize(reimg, (shape[1], shape[0]), cv2.INTER_CUBIC)

        for j in range(stride, shape[1]+stride-patch_size, stride):
            for i in range(stride, shape[0] + stride-patch_size, stride):
                ptc = img[
                    i - stride:i-stride+patch_size, j - stride:j-stride+patch_size]
                reptc = reimg[
                    i - stride:i-stride+patch_size, j - stride:j-stride+patch_size]
                ptc = ptc.astype(float) / 255.
                reptc = reptc.astype(float) / 255.
                data[cnt, :, :, 0] = reptc
                label[cnt, :, :, 0] = ptc[margin:-margin, margin:-margin]
                cnt += 1
    if nmd == 1:
        return data, label, cnt
    else:
        return data, label


if __name__ == "__main__":
    data, label, cnt = load(TRAIN, 1)
    print(cnt)
