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

patch_size = 32
label_size = 20
margin = 6
scale = 2
stride = 20
train_size = 223917#需要先跑一边这个文件，获得测试集大小，再train（写的垃圾qwq
NMD = "SRCNNn" + str(scale) + ".h5"
PATH = "Train/"
TEST="Test/Set5/"
# 2,4:22910  3:21910


def load(qwq, nmd=0):
    cnt = 0
    k = 0
    data = npy.zeros((train_size, patch_size, patch_size, 1), dtype=npy.double)
    label = npy.zeros(
        (train_size, label_size, label_size, 1), dtype=npy.double)
    images_paths = glob.glob(qwq + "*.bmp")
    images_paths += glob.glob(qwq + "*.jpg")
    #images_paths += glob.glob(qwq+"*.png")

    for path in images_paths:
        k = k+1
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        shape = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = img[:, :, 0]

        reimg = cv2.resize(img, (shape[1] // scale, shape[0] // scale))
        reimg = cv2.resize(reimg, (shape[1], shape[0]))

        for j in range(stride, shape[1]+stride-patch_size, stride):
            for i in range(stride, shape[0] + stride-patch_size, stride):
                ptc = img[
                    i - stride:i-stride+patch_size, j - stride:j-stride+patch_size]
                reptc = reimg[
                    i - stride:i-stride+patch_size, j - stride:j-stride+patch_size]
                ptc = ptc.astype(float) / 255.
                reptc = reptc.astype(float) / 255.
                data[cnt, :, :, 0] = cv2.GaussianBlur(reptc,(3,3),0)
                label[cnt, :, :, 0] = ptc[margin:-margin, margin:-margin]
                cnt += 1
    if nmd == 1:
        return data, label, cnt
    else:
        return data, label


if __name__ == "__main__":
    data, label, cnt = load(PATH, 1)
    print(cnt)
