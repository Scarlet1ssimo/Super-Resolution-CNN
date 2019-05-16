# 自己骗自己

如何自己骗自己般地用keras做一个Super Resolution

## 获取数据集

其实自己找也完全没有问题

### 训练与测试

>Specically, the training set consists of 91 images. The Set5 [2] (5 images) is used to evaluate the performance of upscaling factors 2, 3, and 4, and Set14 [28] (14 images) is used to evaluate the upscaling factor 3.

训练图片：91张图片
测试图片：Set5和Set14

### 导入图片

从训练图片中获得训练训练集
详见代码，基本都是opencv2

### 裁剪图片

>The sub-images are extracted from original images with a stride of 14.

先把图片resize小再resize回去，得到低分辨率的图

对每张图片以14px为跨度，33px*33px为大小，裁剪得到训练集

## 预处理

论文：

> To synthesize the low-resolution samples {Yi}, we blur a sub-image by a proper Gaussian kernel, sub-sample it by the upscaling factor, and upscale it by the same factor via bicubic interpolation.

对于sample图还需要高斯模糊。其实是让模型能把模糊的图搞得更清晰，变相增加了学习强度

### 高斯模糊

cv2.GaussianBlur大法好

### bicubic

cv2.resize大法好

## 模型搭建

```py
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
```

### 设置层

根据神仙论文，我们总共有三层模型，全是CNN，详细原理见论文

#### 卷积层1

```py
SRCNN.add(keras.layers.Convolution2D(
        filters=128,
        kernel_size=9,
        strides=1,
        padding='valid',#训练的时候no-padding
        activation='relu',
        input_shape=(imp.patch_size, imp.patch_size, 1)
    ))
```

大致操作：池化，通过9*9的patch来把图形卷成一维向量

#### 卷积层2

```py
SRCNN.add(keras.layers.Convolution2D(
        filters=64,
        kernel_size=3,#说实话，这里效果不明显，配置不太行的建议学论文改为1
        strides=1,
        padding='same',
        activation='relu'
    ))
```

大致操作：把一维向量缩小（加强非线性

#### 卷积层3

```py
SRCNN.add(keras.layers.Convolution2D(
        filters=1,
        kernel_size=5,
        strides=1,
        padding='valid',#训练的时候no-padding
        activation='linear'
    ))
```

大致操作：把一维向量卷回成一个5*5的小patch

keras的卷积层在最后似乎自动把小patch合并成了输出图像，有点nb，不是很懂

### 编译模型

```py
SRCNN.compile(optimizer=keras.optimizers.Adam(lr=0.0003), loss='mean_squared_error',
                  metrics=['mean_squared_error'])
```
因为PSNR是和mse相关的，所以这里用mse作为损失函数和监督

## 训练模型

又到了8说了开冲时间

```py
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
```

学到的技巧：
- `print(SRCNN.summary())`可以检查模型
- 用函数返回数据可以轻松构造数据集和测试集
- 代码中的checkpoint自动保存在测试集上效果最优的模型

还没搞懂callback和batch机制QAQ

## 评估

操作：代码中实现的方法是受padding影响的部分用bicubic填充（也就是把模型跑出来的结果贴到resize后的中间部分。。）

评估也说不出来啥，反正放大的效果挺好，就是不知道为什么Keras会被0.6M的图卡爆内存qwq

## 魔改时间

莫得魔改（素质极差

## 使用方法

### 环境

python3.6 with tensorflow 1.2应该不强求GPU
依赖库缺啥装啥

### 命令

`train.py`：训练

`export1.py [import] [export]`：将`[import]`图片放大成`[export]`文件导出，如`export Origin.jpg Super.jpg`，图片不能太大（其实仅限很小）

