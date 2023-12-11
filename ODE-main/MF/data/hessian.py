import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


torch.random.manual_seed(22)

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 5)

    def forward(self, data):
        x = self.fc1(data)
        x = self.fc2(x)
        # print(self.fc1.weight)
        # print(self.fc1.bias)
        # print(self.fc2.weight)
        # print(self.fc2.bias)

        return x

model = ANN()
for param in model.parameters():
    print(param.size())

data = torch.tensor([1, 2, 3], dtype=torch.float)
label = torch.tensor([1, 1, 5, 7, 8], dtype=torch.float)
pred = model(data)
loss_fn = nn.MSELoss()
loss = loss_fn(pred, label)

grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
print(grads)

hessian_params = []
for k in range(len(grads)):
        hess_params = torch.zeros_like(grads[k])
        print(hess_params)
        for i in range(grads[k].size(0)):
            # 判断是w还是b
            if len(grads[k].size()) == 2:
                # w
                for j in range(grads[k].size(1)):
                    hess_params[i, j] = torch.autograd.grad(grads[k][i][j], model.parameters(), retain_graph=True)[k][i, j]
            else:
                # b
                hess_params[i] = torch.autograd.grad(grads[k][i], model.parameters(), retain_graph=True)[k][i]
        hessian_params.append(hess_params)

print(hessian_params)


