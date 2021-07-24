import os.path

import torch
import torch.nn as nn
from PIL import Image
import PIL.ImageOps
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class CNNModel(nn.Module):

    def __init__(self):
        """
        定义、初始化 模型结构
        """
        super(CNNModel, self).__init__()
        # Convolution 1
        self.cnn1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.relu2 = nn.ReLU()
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        """
        根据init模型 进行一次前向传播计算
        """
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2
        out = self.maxpool2(out)

        # flatten
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        return out


def load_number_data():
    # pandas  Pandas https://pandas.pydata.org/ 是 Python 语言的一个扩展程序库,用于数据分析
    # pd 读取csv中的数据转化为数值
    train = pd.read_csv(r"input/train.csv", dtype=np.float32)
    targets_numpy = train.label.values
    features_numpy = train.loc[:, train.columns != "label"].values
    features_numpy = features_numpy / 255
    return features_numpy, targets_numpy


def show_number(features_numpy, targets_numpy, index=0):
    # 可选，用来辅助认知 input/train.csv 数组和图像的关系
    # Matplotlib Matplotlib 是 Python 的绘图库。 它可与 NumPy 一起使用,提供了一种有效的 MatLab 开源替代方案
    # https://matplotlib.org/
    # plt 用于将像素数组可视化
    plt.imshow(features_numpy[index].reshape(28, 28), cmap='gray')
    plt.title(str(targets_numpy[index]))
    plt.show()
    print(str(targets_numpy[index]))


def test_number():
    """
    测试 显示加载的数据图片
    可选，用来辅助认知 input/train.csv 数组和图像的关系
    """
    features_numpy, targets_numpy = load_number_data()
    show_number(features_numpy, targets_numpy, 3)


def test_train_number_recognition():
    """
    测试 - 训练手写识别
    1、准备数据
    2、准备训练器
    3、训练集训练
    4、测试集测试
    5、保存模型
    """

    def prepare_data():
        # 加载数据
        features_numpy, targets_numpy = load_number_data()

        # 将数据集分割为训练数据，测试数据
        # 按 8：2 拆分 已打标数据 为 训练集 和 测试集
        features_train, features_test, targets_train, targets_test = train_test_split(
            features_numpy, targets_numpy, test_size=0.2, random_state=42
        )

        # 把数据集数组转换成张量 tensor https://zhuanlan.zhihu.com/p/48982978 什么是张量（tensor）
        # 分别对应为 featuresTrain [训练集 图片数据]  targetsTrain [训练集 图片对应的标签]
        featuresTrain = torch.from_numpy(features_train)
        targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
        featuresTest = torch.from_numpy(features_test)
        targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)

        # 训练集 TensorDataset tensor zip  size: 33600
        train_data_set = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
        # 测试集 size: 8400
        test_data_set = torch.utils.data.TensorDataset(featuresTest, targetsTest)
        return train_data_set, test_data_set

    def prepare_train_machine(train_data_set, test_data_set):
        # 模型 CNN
        model = CNNModel()

        # 损失函数， 选择交叉熵损失函数
        # 一文读懂机器学习常用损失函数 https://www.cnblogs.com/guoyaohua/p/9217206.html
        # 关于交叉熵损失函数Cross Entropy Loss https://www.cnblogs.com/jiashun/p/CrossEntropyLoss.html
        loss = nn.CrossEntropyLoss()

        # 学习率 https://www.cnblogs.com/lliuye/p/9471231.html
        learning_rate = 0.1
        # 优化器选择 SGD Optimizer 随机梯度下降 https://www.jiqizhixin.com/graph/technologies/8e284b12-a865-4915-adda-508a320eefde
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # epoch (一波) ， iter（每波下迭代数） 以及 batch size（每次迭代下总次数）
        # 每次iter迭代后，模型权重参数更新一次， 每次epoch后， 全部训练集数据训练完成一次
        # （1）当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一个 epoch。
        # （2）在不能将数据一次性通过神经网络的时候，就需要将数据集分成几个 batch。
        # 例如：比如对于一个有 2000 个训练样本的数据集。将 2000 个样本分成大小为 500 的 batch，那么完成一个 epoch 需要 4 个 iteration。
        # https://www.huaweicloud.com/articles/05aac2f68df788a0bc040729ff43fa99.html
        # 33600
        batch_size = 100
        num_epochs = 20

        train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, model, optimizer, loss, num_epochs

    def train(train_loader, model: CNNModel, optimizer, loss, num_epochs):
        # 全数据跑 epoch = 10 轮
        count = 0
        for epoch in range(num_epochs):
            count = count + 1
            print(f"train: {count}")
            # 每轮迭代， 全数据 sum / batch_size , 每迭代一次，更新一次参数
            for i, (images, labels) in enumerate(train_loader):
                # 梯度清0 https://www.cnblogs.com/sddai/p/14504038.html
                # 根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；
                # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad。
                optimizer.zero_grad()

                # Variable和Tensor本质上没有区别，不过Variable会被放入一个计算图中，然后进行前向传播，反向传播，自动求导。
                # https://geek-docs.com/pytorch/pytorch-tutorial/pytorch-variable.html
                # CNNModel
                outputs = model(Variable(images.view(100, 1, 28, 28)))
                # 根据label、损失函数计算损失
                _loss = loss(outputs, Variable(labels))
                # 自动求导，损失值反向传播
                _loss.backward()
                # 更新一波参数
                optimizer.step()
        return model

    def test(test_loader, model):
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(Variable(images.view(100, 1, 28, 28)))
            predicted = torch.max(outputs.data, 1)[1]
            total += len(labels)
            correct += (predicted == labels).sum()
        accuracy = 100 * correct / float(total)
        print('Total: {}  Correct: {}  Accuracy: {} %'.format(total, correct, accuracy))

    def save(model):
        if not os.path.exists("output"):
            os.makedirs("output")
        PATH = "output/cnn_model.pt"
        torch.save(model.state_dict(), PATH)
        PATH = "output/cnn_model.pkl"
        torch.save(model, PATH)

    def run():
        # 1、准备数据
        train_data_set, test_data_set = prepare_data()
        # 2、准备机器
        train_loader, test_loader, model, optimizer, loss, num_epochs = prepare_train_machine(train_data_set,
                                                                                              test_data_set)
        # 3、机器在训练集开始学习
        model = train(train_loader, model, optimizer, loss, num_epochs)
        # 4、机器在测试集验证、测试学习得到的模型
        test(test_loader, model)
        # 5、保存模型
        save(model)

    run()


def test_predicted():
    """
    测试 - 根据输入图片 输出数字结果
    1、准备数据
    2、准备机器（模型）
    3、预期数据对应的标签（数据结果）
    """

    def prepare_data(_file):
        img = Image.open(_file)
        img = img.resize((28, 28))
        img = img.convert('L')
        img = PIL.ImageOps.invert(img)

        x = np.asarray(img, dtype="int32")
        x = x / 255
        x = x.astype(np.float32)

        feature = x
        feature = torch.from_numpy(feature)
        return feature

    def prepare_machine(_file):
        return torch.load(_file)

    def predict(model: CNNModel, tensor):
        # forward 前向传播 计算变量
        outputs = model(Variable(tensor.view(1, 1, 28, 28)))
        # 使用max()函数对softmax函数的输出值进行操作，求出预测值索引 https://www.jianshu.com/p/3ed11362b54f
        predicted = torch.max(outputs.data, 1)[1]
        # tensor 转 numpy; variable, tensor与numpy区别 https://blog.csdn.net/Asdas_/article/details/105104407
        predicted = predicted.numpy()
        print("predicted: ", predicted[0])

    def run():
        # 1、准备图像数据，转成tensor
        tensor = prepare_data('input/5.png')
        # 2、准备模型
        model = prepare_machine('output/cnn_model.pkl')
        # 3、模型 + 数据 开始计算预测
        predict(model, tensor)

    run()


if __name__ == '__main__':
    test_train_number_recognition()
