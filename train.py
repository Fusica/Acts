import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from Act import *

device = "cuda" if torch.cuda.is_available() else "mps"
dir_path = 'logs'
init_seed = 777
torch.manual_seed(init_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(init_seed)
# 定义训练参数
epochs = 20
batch_size = 1024
learning_rate = 0.01

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
filename = 'output_' + timestamp + '.txt'

if not os.path.exists(dir_path):
    # 如果文件夹不存在，创建文件夹
    os.mkdir(dir_path)
    print('successfully created directory:', dir_path)
else:
    print('directory', dir_path, ' already exists')


class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu1 = DReLU(32)
        self.relu2 = DReLU(64)

    def forward(self, x):
        # 第一个卷积层，使用ReLU作为激活函数，进行2x2的最大池化
        x = self.relu1(self.conv1(x))
        x = F.max_pool2d(x, 2)

        # 第二个卷积层，使用ReLU作为激活函数，进行2x2的最大池化
        x = self.relu2(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # 把数据展平成一维张量，并输入到全连接层1
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # 返回输出
        return F.log_softmax(x, dim=1)


# 创建模型和优化器
model = DigitRecognizer()
torch.compile(model, mode="reduce-overhead")
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 加载MNIST训练数据
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../datasets', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../datasets', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

# 开始训练
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):

        # 定义输入和目标
        data, target = data.to(device), target.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 执行前向传播和反向传播
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 输出日志
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            with open(os.path.join('logs', filename), "a") as f:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                                 len(train_loader.dataset),
                                                                                 100. * batch_idx / len(train_loader),
                                                                                 loss.item()), file=f, flush=True)

    end_time = time.time()
    print('Epoch time: {}'.format(end_time - start_time))
    with open(os.path.join('logs', filename), "a") as f:
        print('Epoch time: {}'.format(end_time - start_time), file=f, flush=True)

    # 在训练循环中计算准确度
    total_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions = output.argmax(dim=1, keepdim=True)
            total_correct += predictions.eq(target.view_as(predictions)).sum().item()
    accuracy = 100. * total_correct / len(test_loader.dataset)
    print('Accuracy: {:.2f}%'.format(accuracy))
    with open(os.path.join('logs', filename), "a") as f:
        print('Accuracy: {:.2f}%'.format(accuracy), file=f, flush=True)
