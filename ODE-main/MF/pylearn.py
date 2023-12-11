import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time


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
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     # 随机生成一个索引
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     # 获取样本及其对应的标签
#     img, label = training_data[sample_idx]
#     # 添加子图
#     figure.add_subplot(rows, cols, i)
#     # 设置标题
#     plt.title(labels_map[label])
#     # 不显示坐标轴
#     plt.axis("off")
#     # 显示灰度图
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


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


torch.random.manual_seed(22)
# 模型定义
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=512),
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
        # print("----------------------------------------------------")
        # grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
        # print(grads)
        # hessian_params = []
        # for k in range(len(grads)):
        #     hess_params = torch.zeros_like(grads[k])
        #     print(hess_params)
        #     for i in range(grads[k].size(0)):
        #         # 判断是w还是b
        #         if len(grads[k].size()) == 2:
        #             # w
        #             for j in range(grads[k].size(1)):
        #                 hess_params[i, j] = \
        #                 torch.autograd.grad(grads[k][i][j], model.parameters(), retain_graph=True)[k][i, j]
        #         else:
        #             # b
        #             hess_params[i] = torch.autograd.grad(grads[k][i], model.parameters(), retain_graph=True)[k][i]
        #     hessian_params.append(hess_params)
        #
        # print(hessian_params)
        # print("----------------------------------------------------")
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
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # 定义模型
    model = NeuralNetwork().to(device)
    # 设置超参数
    learning_rate = 1e-3
    batch_size = 64
    epochs = 50
    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    # 训练模型
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        # if t == epochs-1:
        #     grads = torch.autograd.grad(loss_fn, model.parameters(), retain_graph=True, create_graph=True)
        #     print(grads)
        #     break
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        t_loop(test_dataloader, model, loss_fn, device)
    print("Done!")
    # 保存模型
    torch.save(model.state_dict(), 'model_weights.pth')
    end_time = time.time()
    print("The total running time is {}s.".format(end_time - start_time))


