# -*- coding: utf-8 -*-

import numpy as np
import random
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
from keras.datasets import mnist


class NeuralNet(object):
    # 初始化神经网络，sizes包含了神经网络的层数和每层神经元个数
    def __init__(self, sizes):
        self.sizes_ = sizes
        self.num_layers_ = len(sizes)  # 三层：输入层，一个隐藏层(8个节点), 输出层
        # zip 函数同时遍历两个等长数组的方法
        self.w_ = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  # w_、b_初始化为随机数
        self.b_ = [np.random.randn(y, 1) for y in sizes[1:]]
        # w_是二维数组，w_[0].shape=(8,784), w_[1].shape=(10, 8),权值, 供矩阵乘
        # b_是二维数组，b_[0].shape=(8, 1), b_[1].shape=(10, 1),偏移, 每层间转换的偏移

    # Sigmoid函数，激活函数的一种, 把正负无穷间的值映射到0-1之间
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    # Sigmoid函数的导函数, 不同激活函数导函数不同
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    # 向前传播：已知input，根据w,b算output，用于预测
    def feedforward(self, x):
        for b, w in zip(self.b_, self.w_):
            x = self.sigmoid(np.dot(w, x) + b)
        return x  # 此处的x是0-9每个数字的可能性

    # 单次训练函数，x是本次训练的输入，y是本次训练的实际输出
    # 返回的是需调整的w,b值
    def backprop(self, x, y):
        # 存放待调整的w,b值，nabla是微分算符
        nabla_b = [np.zeros(b.shape) for b in self.b_]  # 与b_大小一样，初值为0
        nabla_w = [np.zeros(w.shape) for w in self.w_]  # 与w_大小一样，初值为0

        activation = x  # 存放层的具体值,　供下层计算
        activations = [x]  # 存储每层激活函数之后的值
        zs = []  # 存放每层激活函数之前的值
        for b, w in zip(self.b_, self.w_):
            z = np.dot(w, activation) + b  # dot是矩阵乘法, w是权值，b是偏移
            zs.append(z)
            activation = self.sigmoid(z)  # 激活函数
            activations.append(activation)

        # 计算输出层的误差，cost_derivative为代价函数的导数
        delta = self.cost_derivative(activations[-1], y) * \
                self.sigmoid_prime(zs[-1])  # 原理见梯度下降部分
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 计算隐藏层的误差
        for l in range(2, self.num_layers_):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.w_[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    # 对每批中的len(mini_batch)个实例，按学习率eta调整一次w,b
    def update_mini_batch(self, mini_batch, eta):
        # 累计调整值
        nabla_b = [np.zeros(b.shape) for b in self.b_]  # 与b_大小一样，值为0
        nabla_w = [np.zeros(w.shape) for w in self.w_]  # 与w_大小一样，值为0
        for x, y in mini_batch:  # 100个值,分别训练
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # eta是预设的学习率(learning rate),用来调节学习的速度. eta越大，调整越大
        # 用新计算出的nable_w调整旧的w_, b_同理
        self.w_ = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.w_, nabla_w)]
        self.b_ = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.b_, nabla_b)]

    # 训练的接口函数
    # training_data是训练数据(x, y);epochs是训练次数;
    # mini_batch_size是每次训练样本数; eta是学习率learning rate
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):  # 用同样数据，训练多次
            random.shuffle(training_data)  # 打乱顺序
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # 把所有训练数据60000个分成每100个/组(mini_batch_size=100)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)  # 分批训练
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    # 计算预测的正确率
    def evaluate(self, test_data):
        # argmax(f(x))是使得 f(x)取得最大值所对应的变量x
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # 代价函数的导数, 对比实际输出与模拟输出的差异, 此时y也是个数组
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    # 预测
    def predict(self, data):
        value = self.feedforward(data)
        return value.tolist().index(max(value))


# 将输入数据转换为神经网络能处理的格式
def load_samples(image, label, dataset="training_data"):
    X = [np.reshape(x, (28 * 28, 1)) for x in image]  # 手写图分辨率28x28
    X = [x / 255.0 for x in X]  # 灰度值范围(0-255)，转换为(0-1)

    # 把y从一个值转成一个数组，对应输出层0-9每个数字出现的概率
    # 5 -> [0,0,0,0,0,1.0,0,0,0];  1 -> [0,1.0,0,0,0,0,0,0,0]
    def vectorized_Y(y):
        e = np.zeros((10, 1))
        e[y] = 1.0
        return e

    if dataset == "training_data":
        Y = [vectorized_Y(y) for y in label]
        pair = list(zip(X, Y))
        return pair
    elif dataset == 'testing_data':
        pair = list(zip(X, label))
        return pair
    else:
        print('Something wrong')


if __name__ == '__main__':
    INPUT = 28 * 28  # 每张图像28x28个像素
    OUTPUT = 10  # 0-9十个分类
    # 784 个输入神经元，一层隐藏层，包含 30 个神经元，输出层包含 10 个神经元
    net = NeuralNet([INPUT, 8, OUTPUT])

    # 从mnist提供的库中装载数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 格式转换
    test_set = load_samples(x_test, y_test, dataset='testing_data')
    train_set = load_samples(x_train, y_train, dataset='training_data')

    # 训练
    net.SGD(train_set, 13, 100, 3.0, test_data=test_set)

    # 计算准确率
    correct = 0;
    for test_feature in test_set:
        if net.predict(test_feature[0]) == test_feature[1]:
            correct += 1
    print("percent: ", correct / len(test_set))
