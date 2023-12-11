import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from loss_plot import semilogy
from k_fold import get_k_fold_data
import time

batch_size = 1024
device = torch.device('cpu')
num_epochs = 200
learning_rate = 0.0006
weight_decay = 0.1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MfDataset(Dataset):
    def __init__(self, u_id, i_id, rating):
        self.u_id = u_id
        self.i_id = i_id
        self.rating = rating

    def __getitem__(self, index):
        return self.u_id[index], self.i_id[index], self.rating[index]

    def __len__(self):
        return len(self.rating)


class MF(nn.Module):
    def __init__(self, num_users, num_items, mean, embedding_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bais = nn.Embedding(num_items, 1)

        self.user_emb.weight.data.uniform_(0, 0.005)  # 0-0.05之间均匀分布
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bais.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([mean]), requires_grad=True)

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bais(i_id).squeeze()
        return (U * I).sum(1) + b_i + b_u + self.mean


def train(model, X_train, y_train, X_valid, y_valid, loss_func, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, valid_ls = [], []

    train_dataset = MfDataset(X_train[:, 0], X_train[:, 1], y_train)
    train_iter = DataLoader(train_dataset, batch_size)

    # 使用Adam优化算法
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model = model.float()
    for epoch in range(num_epochs):
        model.train()  # 如果模型中有Batch Normalization或Dropout层，需要在训练时添加model.train()，使起作用
        total_loss, total_len = 0.0, 0
        for x_u, x_i, y in train_iter:
            #每次循环使用一个batch的数据
            x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            y_pred = model(x_u, x_i)
            l = loss_func(y_pred, y).sum()
            ##if epoch == num_epochs-1:

            # grads = torch.autograd.grad(l, model.parameters(), retain_graph=True, create_graph=True)
            # hessian_params = []
            # for k in range(len(grads)):
            #     hess_params = torch.zeros_like(grads[k])
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
            #     print(hess_params)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            total_loss += l.item()
            total_len += len(y)


        train_ls.append(total_loss / total_len)
        if X_valid is not None:
            model.eval()
            with torch.no_grad():
                n = y_valid.shape[0]
                valid_loss = loss_func(model(X_valid[:, 0], X_valid[:, 1]), y_valid)
            valid_ls.append(valid_loss / n)
        print('epoch %d, train mse %f, valid mse %f' % (epoch + 1, train_ls[-1], valid_ls[-1]))
    return train_ls, valid_ls


def save_model(net):
    PATH = './mv1m_net.pth'
    torch.save(net.state_dict(), PATH)


def load_model():
    PATH = './cifar_net.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.cuda()
    return net


def test(testloader, net):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            #images, labels = data
            images, labels = data[0].cuda(), data[1].cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            _, pred = torch.max(outputs, 1)
            c = (pred == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == "__main__":
    current_time = time.time()
    data = pd.read_csv('ratings.dat', header=None, delimiter='::')
    X, y = data.iloc[:, :2], data.iloc[:, 2]
    # 转换成tensor
    X = torch.tensor(X.values, dtype=torch.int64).to(device)
    y = torch.tensor(y.values, dtype=torch.float32).to(device)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2020)
    mean_rating = data.iloc[:, 2].mean()
    num_users, num_items = max(data[0]) + 1, max(data[1]) + 1
    model = MF(num_users, num_items, mean_rating).to(device)
    loss = torch.nn.MSELoss(reduction="sum")
    train_ls, test_ls = train(model, X_train, y_train, X_test, y_test, loss, num_epochs,
                              learning_rate, weight_decay, batch_size)

    semilogy(range(1, num_epochs + 1), train_ls, "epochs", "mse", range(1, num_epochs + 1), test_ls, ["train", "test"])
    print("\nepochs %d, mean train loss = %f, mse = %f" % (num_epochs, np.mean(train_ls), np.mean(test_ls)))
    print("The total running time is {} minutes {} seconds"
          .format((time.time() - current_time) / 60, (time.time() - current_time) % 60))

    save_model(model)