import random

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import datetime

from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from loss_plot import semilogy
from k_fold import get_k_fold_data
import time
import pdb




# 设置基础参数
batch_size = 1024
device = torch.device('cuda')
num_epochs = 50
learning_rate = 0.0006
weight_decay = 0.1


torch.random.manual_seed(22)


class MfDataset(Dataset):
    def __init__(self, u_id, i_id, rating):
        self.u_id = u_id
        self.i_id = i_id
        self.rating = rating

    def __getitem__(self, index):
        return self.u_id[index], self.i_id[index], self.rating[index]

    def __len__(self):
        return len(self.rating)


# 定义模型
class MF(nn.Module):
    def __init__(self, num_users, num_items, mean, embedding_size=64):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        #self.dream_emb = nn.Embedding(num_users, embedding_size)

        self.user_emb.weight.data.uniform_(0, 0.005)  # 0-0.05之间均匀分布
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)
        #self.dream_emb.weight.data.uniform_(-0.0005, 0.0005)

        # 将不可训练的tensor转换成可训练的类型parameter，并绑定到module里，net.parameter()中就有了这个参数
        self.mean = nn.Parameter(torch.FloatTensor([mean]), requires_grad=True)

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        #U += 0.45*random.uniform(-0.0003, 0.0003)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
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
            # if epoch == num_epochs-1:
            #     cal_IHVP(model, l, y, y_pred, train_dataset)
            #     pdb.set_trace()


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

            # if epoch == num_epochs - 1:
            #     with torch.no_grad():
            #         cal_IHVP(model, l, )
            #     pdb.set_trace()

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


# 训练，k折交叉验证
def train_k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size, num_users, num_items,
                 mean_rating):
    train_l_sum, valid_l_sum = 0.0, 0.0
    loss = torch.nn.MSELoss(reduction="sum").to(device)
    for i in range(k):
        model = MF(num_users, num_items, mean_rating).to(device)
        data = get_k_fold_data(k, i, X_train, y_train)
        train_ls, valid_ls = train(model, *data, loss, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == k:
            semilogy(range(1, num_epochs + 1), train_ls, "epochs", "mse", range(1, num_epochs + 1), valid_ls,
                     ["train", "valid"])
        print('fold %d, train mse %f, valid rmse %f' % (i, train_ls[-1], np.sqrt(valid_ls[-1])))
        print("-------------------------------------------")


def save_model(net):
    PATH = './mv1m_net.pth'
    torch.save(net.state_dict(), PATH)


def cal_IHVP(model, loss, X_train, y_train):
    train_dataset = MfDataset(X_train[:, 0], X_train[:, 1], y_train)
    batch_size_i = len(train_dataset)
    train_iter = DataLoader(train_dataset, batch_size_i)
    # 每次循环使用一个batch的数据
    x_u, x_i, y = X_train[:, 0].to(device), X_train[:, 1].to(device), y_train.to(device)
    y_pred = model(x_u, x_i)
    ls = loss(y_pred, y).sum()
    with torch.no_grad():
        grads = torch.autograd.grad(ls, model.parameters(), retain_graph=True, create_graph=True)
    v_cur_est = grads[1:5]

    i_epoch = 200
    batch_size_j = 16384
    train_iter2 = DataLoader(train_dataset, batch_size_j, shuffle=True)
    for i in range(i_epoch):
        r = np.random.choice(len(X_train[:, 0]), size=[batch_size], replace=False)
        x_u2, x_i2, y2 = X_train[:, 0][r].to(device), X_train[:, 1][r].to(device), y_train[r].to(device)
        # 每次循环使用一个batch的数据
        x_u2, x_i2, y2 = x_u2.to(device), x_i2.to(device), y2.to(device)
        y_pred2 = model(x_u2, x_i2)
        ls2 = loss(y_pred2, y2).sum()
        with torch.no_grad():
            #grads = torch.autograd.grad(ls2, model.parameters(), retain_graph=True)

            parameter_list = nn.ParameterList()
            parameter_list.append(model.user_emb.weight)
            parameter_list.append(model.user_bias.weight)
            parameter_list.append(model.item_emb.weight)
            parameter_list.append(model.item_bias.weight)

            parameter_1_list = []
            parameter_2_list = []
            grads1 = torch.autograd.grad(ls2, parameter_list[0], retain_graph=True, create_graph=True)[0]
            parameter_1_list.append(grads1)
            parameter_2_list.append(grads1)
            grads2 = torch.autograd.grad(ls2, parameter_list[1], retain_graph=True, create_graph=True)[0]
            parameter_1_list.append(grads2)
            parameter_2_list.append(grads2)
            grads3 = torch.autograd.grad(ls2, parameter_list[2], retain_graph=True, create_graph=True)[0]
            parameter_1_list.append(grads3)
            parameter_2_list.append(grads3)
            grads4 = torch.autograd.grad(ls2, parameter_list[3], retain_graph=True, create_graph=True)[0]
            parameter_1_list.append(grads4)
            parameter_2_list.append(grads4)

            for s in range(len(parameter_2_list)):
                temp = v_cur_est[s].clone().detach().requires_grad_(True)
                parameter_2_list[s].mul_(temp)



            inverse_hvp1 = []
            for j in range(len(parameter_2_list)):
                grads_2j = torch.autograd.grad(parameter_2_list[j], parameter_list[j], grad_outputs=
                    torch.ones_like(parameter_2_list[j]), retain_graph=True, create_graph=True)[0]
                inverse_hvp1.append(parameter_1_list[j] + v_cur_est[j] - grads_2j)

            v_cur_est = inverse_hvp1

    return v_cur_est

    # y_pred = torch.tensor(get_y_prediction(model, num_users, num_items), requires_grad=True)
    # y = torch.tensor(get_y_label(X_test, y_test, num_users, num_items))
    # ls = loss(y_pred, y)
    # test = y_pred
    # with torch.no_grad():
    #     grads = torch.autograd.grad(test, y_pred, retain_graph=True, create_graph=True)
    # pdb.set_trace()
    # i_epoch = 10
    # i_batch_size = 16384
    # v_cur_est = grads
    # for j in range(i_epoch):
    #     r = np.random.choice(len(dataset), size=[i_batch_size], replace=False)
    #     pdb.set_trace()

        #hessian_vector_val = hessian_vector_product(loss, model.parameters(), v_cur_est, True)


def cal_influence(model, loss, X_train, y_train, HVPs):
    user = 1
    num = 0
    embedding_influence = []

    while user < 944:
        x_u = []
        x_i = []
        y = []
        result = []
        while num < len(X_train[:, 0]) and user == X_train[:, 0][num]:
            x_u.append(int(X_train[:, 0][num]))
            x_i.append(int(X_train[:, 1][num]))
            y.append(float(y_train[num]))
            num += 1
        x_u, x_i, y = torch.tensor(x_u), torch.tensor(x_i), torch.tensor(y),
        x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
        model.eval()
        y_pred2 = model(x_u, x_i)
        ls2 = loss(y_pred2, y).sum()
        user_grads = torch.autograd.grad(ls2, model.parameters(), retain_graph=True)[1:5]
        params = list(model.parameters())
        for i in range(len(user_grads)):
            result.append((-HVPs[i] * user_grads[i] - params[i + 1]).cpu().detach().numpy())
        embedding_influence.append(result)
        user += 1


def hessian_vector_product(ys, xs, v, parameter_list):
    # Validate the input
    # First backprop
    grads = torch.autograd.grad(ys, xs)
    grads1 = torch.autograd.grad(ys, parameter_list[1])

    v = v.detach()
    # grads = xs
    elemwise_products = [
        torch.mul(grad_elem, v_elem)
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]
    # Second backprop
    seperate = []
    length = len(elemwise_products)-1
    for i in range(length):
        elemwise_products[i+1] = torch.sum(torch.tensor(elemwise_products[i+1], requires_grad=True))
        elemwise_products[i + 1] = torch.tensor(elemwise_products[i+1], requires_grad=True)
        seperate.append(torch.autograd.grad(elemwise_products[i+1], parameter_list[i])[0])
    grads_with_none = seperate

    return_grads = [grad_elem if grad_elem is not None \
                        else torch.zeros(x) \
                    for x, grad_elem in zip(xs, grads_with_none)]
    return return_grads


def calc_influence_single(model, X_train, y_train, test_loader, gpu,
                          recursion_depth, r, s_test_vec=None):
    """Calculates the influences of all training data points on a single
    test dataset image.

    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        gpu: int, identifies the gpu id, -1 for cpu
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated

    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""
    # Calculate s_test vectors if not provided
    if not s_test_vec:
        #晚上修改了，一会儿看这里
        #x_u_test, x_i_test, y_test = test_loader.dataset[0]
        x_u_test, x_i_test, y_test = test_loader.dataset[0:len(test_loader.dataset)]
        x_u_test = test_loader.collate_fn([x_u_test])
        x_i_test = test_loader.collate_fn([x_i_test])
        y_test = test_loader.collate_fn([y_test])
        s_test_vec = calc_s_test_single(model, x_u_test, x_i_test, y_test, X_train, y_train,
                                        gpu, recursion_depth=recursion_depth,
                                        r=r)

    # Calculate the influence function
    train_dataset = MfDataset(X_train[:, 0], X_train[:, 1], y_train)
    train_loader = DataLoader(train_dataset, shuffle=True)
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in range(train_dataset_size):
        x_u, x_i, y = train_loader.dataset[i]
        x_u = train_loader.collate_fn([x_u])
        x_i = train_loader.collate_fn([x_i])
        y = train_loader.collate_fn([y])
        grad_z_vec = grad_z(x_u, x_i, y, model, gpu=gpu)
        tmp_influence = -sum(
            [
                ####################
                # TODO: potential bottle neck, takes 17% execution time
                # torch.sum(k * j).data.cpu().numpy()
                ####################
                torch.sum(k * j).data
                for k, j in zip(grad_z_vec, s_test_vec)
            ]) / train_dataset_size
        pdb.set_trace()
        influences.append(float(tmp_influence))
        print("Calc. influence function: ", i, train_dataset_size)



    # harmful = np.argsort(influences)
    # helpful = harmful[::-1]


    return influences


def calc_s_test_single(model, x_u_test, x_i_test, y_test, X_train, y_train, gpu=-1,
                       damp=0.01, scale=25, recursion_depth=50, r=1):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))

    Arguments:
        model: pytorch model, for which s_test should be calculated
        z_test: test image
        t_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.

    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""
    s_test_vec_list = []
    for i in range(r):
        s_test_vec_list.append(s_test(x_u_test, x_i_test, y_test, model, X_train, y_train,
                                      gpu=gpu, damp=damp, scale=scale,
                                      recursion_depth=recursion_depth))
        print("Averaging r-times: ", i, r)

    ################################
    # TODO: Understand why the first[0] tensor is the largest with 1675 tensor
    #       entries while all subsequent ones only have 335 entries?
    ################################
    s_test_vec = s_test_vec_list[0]
    for i in range(1, r):
        s_test_vec += s_test_vec_list[i]

    s_test_vec = [i / r for i in s_test_vec]

    return s_test_vec


def s_test(x_u_test, x_i_test, y_test, model, X_train, y_train, gpu=-1, damp=0.01, scale=25.0,
           recursion_depth=500):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = grad_z(x_u_test, x_i_test, y_test, model, gpu)
    h_estimate = v.copy()
    loss = torch.nn.MSELoss(reduction="sum")
    train_dataset = MfDataset(X_train[:, 0], X_train[:, 1], y_train)

    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    count = 0
    for i in range(recursion_depth):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        #########################
        # TODO: do x, t really have to be chosen RANDOMLY from the train set?
        #########################
        z_loader = DataLoader(train_dataset, batch_size=16384, shuffle=True)


        for x_u, x_i, y in z_loader:

            count2 = 0
            for i in range(len(h_estimate)):
                h_estimate[i] = h_estimate[i].detach()
            if gpu >= 0:
                x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            y_pred = model(x_u, x_i)
            ls = loss(y_pred, y)
            params = [p for p in model.parameters() if p.requires_grad]
            s1 = time.time()
            print("hvp on count={}.{} forward".format(count, count2))
            hv = hvp(ls, params, h_estimate)
            print("hvp on count={}.{} backward".format(count, count2))
            s2 = time.time()
            print(s2-s1)
            # Recursively caclulate h_estimate
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            count2 += 1
            break
        print("Calc. s_test recursions: ", i, recursion_depth)
        count += 1
    return h_estimate


def grad_z(x_u, x_i, y, model, gpu=-1):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()
    loss = torch.nn.MSELoss(reduction="sum")
    x_u, x_i, y = x_u.flatten(0), x_i.flatten(0), y.flatten(0)

    # initialize
    if gpu >= 0:
        x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
    y_pred = model(x_u, x_i)
    ls = loss(y_pred, y).sum()
    # Compute sum of gradients from model parameters to loss
    params = [p for p in model.parameters() if p.requires_grad]
    return list(torch.autograd.grad(ls, params, create_graph=True))


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = torch.autograd.grad(y, w, retain_graph=True, create_graph=True)


    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = torch.autograd.grad(elemwise_products, w, create_graph=True)

    return return_grads


def get_y_prediction(model, num_users, num_items):
    prediction = []
    item_tensor = torch.arange(1, num_items).to(device)
    user_id = 1
    while user_id < num_users:
        user_tensor = torch.full((num_items-1,), user_id).to(device)
        prediction_ui = model(user_tensor, item_tensor).tolist()
        user_id += 1
        prediction.append(prediction_ui)
    return prediction
    # for users in range(num_users):
    #     user_id = torch.tensor([users]).to(device)
    #     temp = []
    #     for items in range(num_items):
    #         item_id = torch.tensor([items]).to(device)
    #         prediction_ui = model(user_id, item_id)
    #         temp.append(float(prediction_ui))
    #     prediction.append(temp)


def get_y_label(X_test, y_test, num_users, num_items):
    cols = num_items-1
    rows = num_users-1
    label_list = [[0] * cols for _ in range(rows)]
    for i in range(len(X_test)):
        x_u = int(X_test[i][0])-1
        y_i = int(X_test[i][1])-1
        label_list[x_u][y_i] = int(y_test[i])
    return label_list


def recommend_metrics(prediction, label, top_k):
    scores = torch.tensor(prediction)
    labels = torch.tensor(label)
    _, indices = torch.topk(scores, k=top_k, dim=1)  # 获取每个用户的前 K 个预测值的索引
    predicted_labels = torch.zeros_like(labels)  # 创建与真实标签相同大小的零矩阵，用于存储预测结果
    predicted_labels.scatter_(1, indices, 1)  # 将预测值对应的位置设置为 1
    correct = torch.sum(predicted_labels * labels, dim=1)  # 计算每个用户的命中数量
    hr = torch.mean(correct.float())  # 计算平均命中率

    # _, sorted_indices = torch.sort(scores, descending=True, dim=1)  # 将预测值降序排列
    # sorted_labels = torch.gather(labels, 1, sorted_indices)  # 根据排序后的索引获取真实标签
    # dcg = torch.sum(sorted_labels / torch.log2(torch.arange(2, top_k + 2).float()), dim=1)  # 计算折损累积增益
    # ideal_sorted_labels, _ = torch.sort(labels, descending=True, dim=1)  # 将真实标签按照最佳排列顺序排序
    # ideal_dcg = torch.sum(ideal_sorted_labels / torch.log2(torch.arange(2, top_k + 2).float()), dim=1)  # 计算最佳折损累积增益
    # ndcg = torch.mean(dcg / ideal_dcg)  # 计算平均归一化折损累积增益

    print("Hit:", hr.item())
    # print("NDCG:", ndcg.item())


def cal_MSE(prediction, X_test, y_test):
    total_SE = 0
    count = 0
    for pairs in range(len(X_test)):
        x_u = int(X_test[pairs][0])-1
        y_i = int(X_test[pairs][1])-1
        y_prediction = prediction[x_u][y_i]
        total_SE += (y_prediction - float(y_test[pairs])) * (y_prediction - float(y_test[pairs]))
        count += 1
    handmade_MSE = float(total_SE) / float(count)
    print("count = {}".format(count))
    print("handmade SE = {}".format(handmade_MSE))


def main():
    current_time = time.time()
    # 加载数据
    extra_data = pd.read_csv('extra_users/good_users/good_20.csv', header=None)
    train_data = pd.read_csv('../dataset/ml-100k/ml-100k.train.rating', header=None, delimiter='\t')
    test_data = pd.read_csv('../dataset/ml-100k/ml-100k.test.rating', header=None, delimiter='\t')
    #train_data = train_data.append(extra_data)
    X_train, y_train = train_data.iloc[:, :2], train_data.iloc[:, 2]
    bad_rating_list = np.load("../dataset/ml-100k/bad_ratings.npy")
    count = 0
    # for i in range(0):
    #     df = X_train[(X_train[0]==bad_rating_list[i][0])&(X_train[1]==bad_rating_list[i][1])]
    #     if not df.empty:
    #         count += 1
    #         row = X_train[(X_train[0]==bad_rating_list[i][0])&(X_train[1]==bad_rating_list[i][1])].index[0]
    #         X_train = X_train.drop(row)
    #         y_train = y_train.drop(row)
            #y_train.iloc[row] = 3
        # else:
        #     X_train = X_train.append(pd.DataFrame([[bad_rating_list[i][0],
        #                                             bad_rating_list[i][1]]], columns=X_train.columns))
        #
        #     y_train = y_train.append(pd.Series([5]))
    # print(count)
    X_test, y_test = test_data.iloc[:, :2], test_data.iloc[:, 2]

    # 转换成tensor
    X_train = torch.tensor(X_train.values, dtype=torch.int64).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32, requires_grad=True).to(device)
    X_test = torch.tensor(X_test.values, dtype=torch.int64).to(device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32, requires_grad=True).to(device)

    # 划分训练集和测试集
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2020)
    mean_rating = train_data.iloc[:, 2].mean()
    num_users, num_items = max(train_data[0]) + 1, max(train_data[1]) + 1

    # Train_data = pd.read_csv('../dataset/filmtrust/filmtrust.train.rating', header=None, delimiter='\t')
    # Test_data = pd.read_csv('../dataset/filmtrust/filmtrust.test.rating', header=None, delimiter='\t')
    #
    # X_train, y_train = Train_data.iloc[:, :2], Train_data.iloc[:, 2]
    # X_test, y_test = Test_data.iloc[:, :2], Test_data.iloc[:, 2]
    # X_train = torch.tensor(X_train.values, dtype=torch.int64).to(device)
    # y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    # X_test = torch.tensor(X_test.values, dtype=torch.int64).to(device)
    # y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)
    # mean_rating = Train_data.iloc[:, 2].mean()
    # num_users, num_items = max(Train_data[0]) + 1, max(Train_data[1]) + 1

    # 交叉验证选择最优超参数
    # train_k_fold(8, X_train, y_train, num_epochs=num_epochs, learning_rate=learning_rate,
    #              weight_decay=weight_decay, batch_size=batch_size, num_users=num_users, num_items=num_items,
    #              mean_rating=mean_rating)

    model = MF(num_users, num_items, mean_rating).to(device)
    loss = torch.nn.MSELoss(reduction="sum")
    train_ls, test_ls = train(model, X_train, y_train, X_test, y_test, loss, num_epochs,
                              learning_rate, weight_decay, batch_size)

    #cal_Inf_v2
    model.eval()
    train_dataset = MfDataset(X_train[:, 0], X_train[:, 1], y_train)
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_dataset = MfDataset(X_test[:, 0], X_test[:, 1], y_test)
    test_loader = DataLoader(test_dataset, shuffle=True)
    recursion_depth = 1000
    gpu = 0
    r = 1
    influences = calc_influence_single(model, X_train, y_train, test_loader, gpu, recursion_depth, r, s_test_vec=None)
    np.save('pairs_influences.npy', influences)

    # model.eval()
    # prediction = get_y_prediction(model, num_users, num_items)
    # cal_MSE(prediction, X_test, y_test)
    # label = get_y_label(X_test, y_test, num_users, num_items)
    # recommend_metrics(prediction, label, 10)
    # HVPs = cal_IHVP(model, loss, X_train, y_train)


    #np.save('embedding_influences.npy', embedding_influence)

    #semilogy(range(1, num_epochs + 1), train_ls, "epochs", "mse", range(1, num_epochs + 1), test_ls, ["train", "test"])

    print("\nepochs %d, mean train loss = %f, mse = %f" % (num_epochs, sum(train_ls)/len(train_ls),
                                                            sum(test_ls)/len(test_ls)))
    print("The total running time is {} minutes {} seconds"
          .format(int((time.time()-current_time)/60), (time.time()-current_time)%60))

    save_model(model)

if __name__ == '__main__':
    main()
