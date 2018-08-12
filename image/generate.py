# coding:utf-8
import os
from keras.models import Input, Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import glob
import cv2
import random
import numpy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""    ###这两句代码用来关闭CUDA加速

img_h, img_w, n_channels = 128, 128, 3  # 图像高，宽，通道数
classes = 5  # 类别
train_batch_size = 40  # 训练batch大小
test_batch_size = 20   # 测试batch大小，本次实验用作验证集
train_path = './all_data/train/*'
test_path = './all_data/test/*'


# 得到one-hot编码   也可用keras自带函数np_utils.to_categorical()
def get_one_hot(num, classes):
    one_hot = numpy.zeros(shape=(1, classes))
    one_hot[:, num] = 1
    return one_hot


# 数据生成器，可以实时对数据做变化，可用于数据集比较大的情况
class DataGenerate:
    def __init__(self, img_path, img_h, img_w, n_channels, batch_size=1, flip=True):
        self.img_h = img_h
        self.img_w = img_w
        self.n_channels = n_channels
        self.flip = flip    # 是否做水平或者垂直翻转
        self.all_path = glob.glob(img_path)
        self.all_num = len(self.all_path)
        self.index = 0
        self.batch_size = batch_size

    # 得到数据和对应的label
    def get_batch(self, index, size):
        all_data = numpy.zeros(shape=(size, self.img_h, self.img_w, self.n_channels), dtype='float32')
        all_label = numpy.zeros(shape=(size, classes))
        for i, img_name in enumerate(self.all_path[index:index+size]):
            img = cv2.imread(img_name, 1)
            # print(img_name)
            img = cv2.resize(img, dsize=(img_w, img_h))  # 归一化图片输入尺寸
            img = img - 128.0  # 数据归一到【-1,1】
            img = img / 128.0
            if self.flip:
                p = numpy.random.random()
                if p > 0.5:
                    p_v_h = numpy.random.random()
                    if p_v_h > 0.5:
                        img = img[:, ::-1, :]  # 水平翻转
                    else:
                        img = img[::-1, :, :]  # 垂直翻转
            all_data[i, :, :, :] = img
            label = int(list(img_name)[-7])-3
            all_label[i, :] = get_one_hot(label, classes)  # 获取label  one-hot编码
        return all_data, all_label

    # python生成器，获取数据和label
    def next_data(self):
        while 1:
            data_label = self.get_batch(self.index, self.batch_size)
            self.index += self.batch_size
            if self.index > self.all_num:
                self.index = 0
                random.shuffle(self.all_path)
            yield data_label


if __name__ == '__main__':
    train_data = DataGenerate(train_path, img_h, img_w, n_channels, 40)  # 定义生成器
    test_data = DataGenerate(test_path, img_h, img_w, n_channels, 20, flip=False)
    input_data = Input(shape=(img_h, img_w, n_channels))
    out = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_data)
    out = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
    out = BatchNormalization(axis=3)(out)
    out = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(out)
    out = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
    out = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
    out = BatchNormalization(axis=3)(out)
    out = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(out)
    out = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
    out = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
    out = BatchNormalization(axis=3)(out)
    out = GlobalAveragePooling2D()(out)
    # out = Dense(units=512,activation='relu')(out) # 数据集比较小，可不加这一层
    out = Dense(units=5, activation='softmax')(out)
    model = Model(inputs=input_data, outputs=out)
    model.summary()
    model_check = ModelCheckpoint('./model.h5',
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='auto',
                                  period=1)
    # 用来验证每一epoch是否是最好的模型用来保存  val_loss
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_data.next_data(), train_data.all_num//train_batch_size, 100, callbacks=[model_check],
                        validation_data=test_data.next_data(), validation_steps=test_data.all_num//test_batch_size)
    model.save('./final_model.h5')











