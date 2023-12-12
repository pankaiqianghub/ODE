import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 训练数据集
training_data = datasets.FashionMNIST(
    root="data",  # 数据集下载路径
    train=True,  # True为训练集，False为测试集
    download=True,  # 是否要下载
    transform=ToTensor()  # 对样本数据进行处理，转换为张量数据
)
# 测试数据集
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# 标签字典，一个key键对应一个label
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
# 设置画布大小
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    # 随机生成一个索引
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    # 获取样本及其对应的标签
    img, label = training_data[sample_idx]
    # 添加子图
    figure.add_subplot(rows, cols, i)
    # 设置标题
    plt.title(labels_map[label])
    # 不显示坐标轴
    plt.axis("off")
    # 显示灰度图
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# 训练数据加载器
train_dataloader = DataLoader(
    dataset=training_data,
    # 设置批量大小
    batch_size=64,
    # 打乱样本的顺序
    shuffle=True)
# 测试数据加载器
test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=True
)


# 模型定义
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 优化模型参数
def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # 前向传播，计算预测值
        pred = model(X)
        # 计算损失
        loss = loss_fn(pred, y)
        # 反向传播，优化参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# 测试模型性能
def t_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            # 前向传播，计算预测值
            pred = model(X)
            # 计算损失
            test_loss += loss_fn(pred, y).item()
            # 计算准确率
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    print(training_data[0])