# -*- coding: utf-8 -*-
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128  # 批尺寸
num_classes = 10 # 0-9十个数字对应10个分类
epochs = 12  # 训练12次

# input image dimensions
img_rows, img_cols = 28, 28  # 训练图片大小28x28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 下面model相关的是关键部分
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),  # 加卷积层，核大小3x3，输出维度32
                 activation='relu',  # 激活函数为relu
                 input_shape=input_shape))  # 传入数据
model.add(Conv2D(64, (3, 3), activation='relu')) # 又加一个卷积层
model.add(MaxPooling2D(pool_size=(2, 2)))  # 以2x2为一块池化
model.add(Dropout(0.25))  # 随机断开25%的连接
model.add(Flatten())  # 扁平化，例如将28x28x1变为784的格式
model.add(Dense(128, activation='relu'))  # 加入全连接层
model.add(Dropout(0.5))  # 再加一层Dropout
model.add(Dense(num_classes, activation='softmax'))  # 加入到输出层的全连接

model.compile(loss=keras.losses.categorical_crossentropy,  # 设损失函数
              optimizer=keras.optimizers.Adadelta(),  # 设学习率
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,  # 设批大小
          epochs=epochs,  # 设学习次数
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)  # 用测试集评测
print('Test loss:', score[0])
print('Test accuracy:', score[1])
