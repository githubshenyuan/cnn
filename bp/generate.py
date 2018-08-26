import numpy as np
from keras.datasets import mnist
import numpy as np
import random


class Network(object):
    def __init__(self, sizes):
        # 神经网络的层数
        # 784 个输入神经元，一层隐藏层，包含 30 个神经元，输出层包含 10 个神经元
        self.num_layers_ = len(sizes)
        # 除去输入层，随机产生每层中 y 个神经元的 biase 值（0 - 1）
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # 随机产生每条连接线的 weight 值（0 - 1）
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def backprop(self, x, y):
        """
        :param x:
        :param y:
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 前向传输
        activation = x
        # 储存每层的神经元的值的矩阵，下面循环会 append 每层的神经元的值
        activations = [x]
        # 储存每个未经过 sigmoid 计算的神经元的值
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 求 δ 的值
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        # 乘于前一层的输出值
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            # 从倒数第 **l** 层开始更新，**-l** 是 python 中特有的语法表示从倒数第 l 层开始计算
            # 下面这里利用 **l+1** 层的 δ 值来计算 **l** 的 δ 值
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, eta):
        # 根据 biases 和 weights 的行列数创建对应的全部元素值为 0 的空矩阵
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:  # 100个值,分别训练
            # 根据样本中的每一个输入 x 的其输出 y，计算 w 和 b 的偏导数
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # eta是预设的学习率(learning rate),用来调节学习的速度. eta越大，调整越大
        # 用新计算出的nable_w调整旧的w_, b_同理
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.w_, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.b_, nabla_b)]

    """
    随机梯度下降
    :param training_data: 输入的训练集
    :param epochs: 迭代次数
    :param mini_batch_size: 小样本数量
    :param eta: 学习率
    :param test_data: 测试数据集
    """
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            # 搅乱训练集，让其排序顺序发生变化
            random.shuffle(training_data)
        # 把所有训练数据60000个分成每100个/组(mini_batch_size=100)
        mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
        # 分批训练
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, eta)

def sigmoid(z):
    """
    求 sigmoid 函数的值
    :param z:
    :return:
    """
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """
    求 sigmoid 函数的导数
    :param z:
    :return:
    """
    return sigmoid(z)*(1-sigmoid(z))


if __name__ == '__main__':
    print("结束")





