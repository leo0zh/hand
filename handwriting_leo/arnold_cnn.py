import PIL.ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # build network
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # [32, h, w]
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)  # [64, h/2, w/2]
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)  # [128, h/4, w/4]
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # [n, 32, 14, 14]

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # [n, 64, 7, 7]

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)  # [n, 128, 3, 3]

        n, c, h, w = x.size()
        x = x.view(n, -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

    def loss(self, x, labels):
        loss = F.cross_entropy(x, labels)
        return loss


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
        model = Network()
        model = model.to(torch.device('cpu'))
        model.load_state_dict(torch.load(_file, map_location=torch.device('cpu')))
        return model

    def predict(model: Network, tensor):
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
        model = prepare_machine('output/c97a0a6be3d6f4b0d00b2d96b534a40b.pth')
        # 3、模型 + 数据 开始计算预测
        predict(model, tensor)

    run()
