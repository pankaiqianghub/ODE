import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import time
from torch.utils.data import DataLoader, Dataset, TensorDataset
import argparse
from scipy.sparse import dok_matrix
import pdb
import torch.nn.functional as F





torch.random.manual_seed(22)

def parse_args():
    parser = argparse.ArgumentParser(description="Command line arguments")

    parser.add_argument('--regularization', type=float, default=1e-12,
                        help="Set regularization value")
    parser.add_argument('--batch_size', type=int, default=1024,
                        help="Set batch size value")
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help="Set device (cuda or cpu)")
    parser.add_argument('--num_epochs', type=int, default=10,
                        help="Set number of epochs")
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help="Set learning rate value")
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help="Set weight decay value")

    args = parser.parse_args()
    return args
args = parse_args()
regularization = args.regularization
batch_size = args.batch_size
device = torch.device(args.device)
num_epochs = args.num_epochs
learning_rate = args.learning_rate
weight_decay = args.weight_decay


class MfDataset(Dataset):
    def __init__(self, u_id, i_id, rating):
        self.u_id = u_id
        self.i_id = i_id
        self.rating = rating

    def __getitem__(self, index):
        return self.u_id[index], self.i_id[index], self.rating[index]

    def __len__(self):
        return len(self.rating)


class FunkSVD(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(FunkSVD, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.user_embedding.weight.data.uniform_(0, 0.005)
        self.item_embedding.weight.data.uniform_(0, 0.005)

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        dot_product = torch.sum(torch.mul(user_embeds, item_embeds), dim=1)
        preds = dot_product

        regularization_term = regularization * (
                torch.norm(user_embeds) + torch.norm(item_embeds)
        )
        preds += regularization_term

        return preds


def train(model, X_train, y_train, X_valid, y_valid, loss_func, num_epochs, learning_rate, weight_decay, batch_size,
          X_test, y_test, num_users, num_items):
    train_ls, valid_ls = [], []

    train_dataset = MfDataset(X_train[:, 0], X_train[:, 1], y_train)
    train_iter = DataLoader(train_dataset, batch_size)

    # 使用Adam优化算法
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model = model.float()
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_len = 0.0, 0
        for x_u, x_i, y in train_iter:
            x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            y_pred = model(x_u, x_i)
            l = loss_func(y_pred, y).sum()
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
        model.eval()
        prediction = get_y_prediction(model, num_users, num_items)
        label = get_y_label(X_test, y_test, num_users, num_items)
        recommend_metrics(prediction, label, 10)
    return train_ls, valid_ls


def calc_influence_single(model, X_train, y_train, test_loader, gpu,
                          recursion_depth, r, s_test_vec=None):
    # Calculate s_test vectors if not provided
    if not s_test_vec:
        x_u_test, x_i_test, y_test = test_loader.dataset[0:len(test_loader.dataset)]
        x_u_test = test_loader.collate_fn([x_u_test])
        x_i_test = test_loader.collate_fn([x_i_test])
        y_test = test_loader.collate_fn([y_test])
        s_test_vec = calc_s_test_single(model, x_u_test, x_i_test, y_test, X_train, y_train,
                                        gpu, recursion_depth=recursion_depth,
                                        r=r)

    # Calculate the influence function
    train_dataset = MfDataset(X_train[:, 0], X_train[:, 1], y_train)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
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
                torch.sum(k * j).data
                for k, j in zip(grad_z_vec, s_test_vec)
            ]) / train_dataset_size
        influences.append(float(tmp_influence))
        #print("Calc. influence function: ", i, train_dataset_size)
    harmful = np.argsort(influences)
    np.save('pairs_influences_rank_100k_919.npy', harmful)
    helpful = harmful[::-1]
    return influences, harmful, helpful


def calc_s_test_single(model, x_u_test, x_i_test, y_test, X_train, y_train, gpu=-1,
                       damp=0.01, scale=25, recursion_depth=50, r=1):
    s_test_vec_list = []
    for i in range(r):
        s_test_vec_list.append(s_test(x_u_test, x_i_test, y_test, model, X_train, y_train,
                                      gpu=gpu, damp=damp, scale=scale,
                                      recursion_depth=recursion_depth))
        #print("Averaging r-times: ", i, r)
    s_test_vec = s_test_vec_list[0]
    for i in range(1, r):
        s_test_vec += s_test_vec_list[i]

    s_test_vec = [i / r for i in s_test_vec]

    return s_test_vec


def s_test(x_u_test, x_i_test, y_test, model, X_train, y_train, gpu=-1, damp=0.01, scale=25.0,
           recursion_depth=500):
    v = grad_z(x_u_test, x_i_test, y_test, model, gpu)
    h_estimate = v.copy()
    loss = torch.nn.MSELoss(reduction="mean")
    train_dataset = MfDataset(X_train[:, 0], X_train[:, 1], y_train)

    count = 0
    for i in range(recursion_depth):
        s1 = time.time()
        # print("hvp on count={} forward".format(count))
        z_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        for x_u, x_i, y in z_loader:
            for i in range(len(h_estimate)):
                h_estimate[i] = h_estimate[i].detach()
            if gpu >= 0:
                x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            y_pred = model(x_u, x_i)
            ls = loss(y_pred, y)
            params = [p for p in model.parameters() if p.requires_grad]
            hv = hvp(ls, params, h_estimate)
            # Recursively caclulate h_estimate
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
        s2 = time.time()
        # print(s2 - s1)
        # print(h_estimate)
        #print("Calc. s_test recursions: ", i, recursion_depth)
        count += 1
    return h_estimate


def hvp(y, w, hv):
    if len(w) != len(hv):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = torch.autograd.grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products=0
    for grad_elem, v_elem in zip(first_grads, hv):
        elemwise_products += torch.sum(grad_elem * v_elem)
        #elemwise_products.append(grad_elem * v_elem)

    # Second backprop
    # seperate = []
    # for i in range(len(elemwise_products)):
    #     seperate.append(torch.autograd.grad(elemwise_products[i], w, create_graph=True,
    #                                    grad_outputs=torch.ones_like(elemwise_products[i])))
    return_grads = torch.autograd.grad(elemwise_products, w, create_graph=True)

    return return_grads


def grad_z(x_u, x_i, y, model, gpu=-1):
    model.eval()
    loss = torch.nn.MSELoss(reduction="mean")
    x_u, x_i, y = x_u.flatten(0), x_i.flatten(0), y.flatten(0)

    # initialize
    if gpu >= 0:
        x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
    y_pred = model(x_u, x_i)
    ls = loss(y_pred, y).sum()
    # Compute sum of gradients from model parameters to loss
    params = [p for p in model.parameters() if p.requires_grad]
    return list(torch.autograd.grad(ls, params, create_graph=True))


def choose_dataset(X_train, y_train, proportion=0.03, method='bad'):
    harmful = np.load('pairs_influences_rank_mv1m.npy')
    delete_length = int(len(X_train)*proportion)
    if method == 'good':
        delete_list = harmful[0: delete_length]
        X_train = X_train[~X_train.index.isin(delete_list)]
        y_train = y_train[~y_train.index.isin(delete_list)]

    elif method == 'bad':
        helpful = harmful[::-1]
        delete_list = helpful[0: delete_length]
        X_train = X_train[~X_train.index.isin(delete_list)]
        y_train = y_train[~y_train.index.isin(delete_list)]

    return X_train, y_train


def cal_similarity(user_embedding, item_embedding):
    user_embedding = torch.tensor(user_embedding)
    item_embedding = torch.tensor(item_embedding)
    user_similarity_mat = []
    item_similarity_mat = []
    for user_i in range(len(user_embedding)):
        i_user_similarity_list = []
        for user_j in range(len(user_embedding)):
            if user_i == user_j:
                i_user_similarity_list.append(0)
            else:
                cos_sim = user_embedding[user_i].dot(user_embedding[user_j]) / (
                        np.linalg.norm(user_embedding[user_i]) * np.linalg.norm(user_embedding[user_j]))
                i_user_similarity_list.append(-cos_sim)
        user_sim = np.argsort(i_user_similarity_list)[0:100]
        user_similarity_mat.append(user_sim)
    for item_i in range(len(item_embedding)):
        i_item_similarity_list = []
        for item_j in range(len(item_embedding)):
            if item_i == item_j:
                i_item_similarity_list.append(0)
            else:
                cos_sim = item_embedding[item_i].dot(item_embedding[item_j]) / (
                        np.linalg.norm(item_embedding[item_i]) * np.linalg.norm(item_embedding[item_j]))
                i_item_similarity_list.append(-cos_sim)
        item_sim = np.argsort(i_item_similarity_list)[0:100]
        item_similarity_mat.append(item_sim)
    np.save('user_similarity_mat_100k_919.npy', user_similarity_mat)
    np.save('user_similarity_mat_100k_919.npy', item_similarity_mat)
    return user_similarity_mat, item_similarity_mat


def add_ratings(X_train, y_train, user_similarity_mat, item_similarity_mat, influences, X_train_extra, y_train_extra):
    add_x_matrix = []
    add_y_matrix = []
    for i in range(int(len(X_train)*0.03)):
        origin_u = int(X_train_extra[influences[i]][0])
        origin_i = int(X_train_extra[influences[i]][1])
        origin_r = int(y_train_extra[influences[i]])
        new_u = user_similarity_mat[origin_u][0]
        new_i = item_similarity_mat[origin_i][0]
        add_x_matrix.append([new_u, new_i])
        add_y_matrix.append(origin_r)
    add_x_matrix = torch.tensor(add_x_matrix).to(device)
    add_y_matrix = torch.tensor(add_y_matrix).to(device)
    X_train = torch.cat((X_train, add_x_matrix), 0)
    y_train = torch.cat((y_train, add_y_matrix), 0)

    return X_train, y_train


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
    print("Hit:", hr.item())

    sorted_labels = torch.gather(labels, 1, indices)  # 根据排序后的索引获取真实标签
    discounts = torch.log2(torch.arange(2, top_k + 2).float().to(scores.device))  # 折扣因子
    dcg = torch.sum(sorted_labels / discounts, dim=1)  # 计算折损累积增益
    ideal_sorted_labels, _ = torch.sort(labels, descending=True, dim=1)  # 将真实标签按照最佳排列顺序排序
    ideal_sorted_labels = ideal_sorted_labels[:, :top_k]  # 获取最佳排列的前k个标签
    ideal_dcg = torch.sum(ideal_sorted_labels / discounts, dim=1)  # 计算最佳折损累积增益
    ndcg = torch.mean(dcg / ideal_dcg)  # 计算平均归一化折损累积增益
    print("NDCG:", ndcg.item())


def main():
    current_time = time.time()
    # 加载数据
    train_data = pd.read_csv('../dataset/ml-100k/ml-100k.train.rating', header=None, delimiter='\t')
    test_data = pd.read_csv('../dataset/ml-100k/ml-100k.test.rating', header=None, delimiter='\t')
    X_train, y_train = train_data.iloc[:, :2], train_data.iloc[:, 2]
    #X_train, y_train = choose_dataset(X_train, y_train)
    X_test, y_test = test_data.iloc[:, :2], test_data.iloc[:, 2]

    # 转换成tensor
    X_train_o = torch.tensor(X_train.values, dtype=torch.int64).to(device)
    y_train_o = torch.tensor(y_train.values, dtype=torch.float32, requires_grad=True).to(device)
    X_test = torch.tensor(X_test.values, dtype=torch.int64).to(device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32, requires_grad=True).to(device)

    # 划分训练集和测试集
    num_users, num_items = max(train_data[0]) + 1, max(train_data[1]) + 1

    model = FunkSVD(num_users, num_items).to(device)
    loss = torch.nn.MSELoss(reduction="sum")
    train_ls, test_ls = train(model, X_train_o, y_train_o, X_test, y_test, loss, num_epochs,
                              learning_rate, weight_decay, batch_size, X_test, y_test, num_users, num_items)

    user_embedding_list = []
    item_embedding_list = []
    for i in range(len(model.user_embedding.weight)):
        user_embedding_list.append(model.user_embedding.weight[i].tolist())
    for j in range(len(model.item_embedding.weight)):
        item_embedding_list.append(model.item_embedding.weight[j].tolist())
    np.save('user_embedding_list_100k_919.npy', user_embedding_list)
    np.save('item_embedding_list_100k_919.npy', item_embedding_list)
    #user_similarity_mat, item_similarity_mat = cal_similarity(user_embedding_list, item_embedding_list)

    train_dataset = MfDataset(X_train_o[:, 0], X_train_o[:, 1], y_train_o)
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_dataset = MfDataset(X_test[:, 0], X_test[:, 1], y_test)
    test_loader = DataLoader(test_dataset, shuffle=True)
    recursion_depth = 10
    gpu = 0
    r = 1
    influences, harmful, helpful = calc_influence_single(model, X_train_o, y_train_o, test_loader, gpu, recursion_depth, r, s_test_vec=None)
    X_train_extra, y_train_extra = X_train_o, y_train_o

    # countx = 0
    # for final_epoch in range(10):
    #     print("---------------start bad epoch = {}-------------------".format(countx))
    #     X_train_new, y_train_new = add_ratings(X_train_o, y_train_o, user_similarity_mat,
    #                                            item_similarity_mat, harmful, X_train_extra, y_train_extra)
    #
    #     model2 = FunkSVD(num_users, num_items).to(device)
    #     loss = torch.nn.MSELoss(reduction="sum")
    #     train_ls, test_ls = train(model2, X_train_new, y_train_new, X_test, y_test, loss, num_epochs,
    #                               learning_rate, weight_decay, batch_size)
    #
    #     user_embedding_list = []
    #     item_embedding_list = []
    #     for i in range(len(model2.user_embedding.weight)):
    #         user_embedding_list.append(model2.user_embedding.weight[i].tolist())
    #     for j in range(len(model2.item_embedding.weight)):
    #         item_embedding_list.append(model2.item_embedding.weight[j].tolist())
    #     user_similarity_mat, item_similarity_mat = cal_similarity(user_embedding_list, item_embedding_list)
    #
    #     model2.eval()
    #     train_dataset = MfDataset(X_train_new[:, 0], X_train_new[:, 1], y_train_o)
    #     train_loader = DataLoader(train_dataset, shuffle=True)
    #     test_dataset = MfDataset(X_test[:, 0], X_test[:, 1], y_test)
    #     test_loader = DataLoader(test_dataset, shuffle=True)
    #     recursion_depth = 150
    #     gpu = 0
    #     r = 1
    #     influences, harmful, helpful = calc_influence_single(model2, X_train_new, y_train_new, test_loader, gpu,
    #                                                          recursion_depth, r, s_test_vec=None)
    #     X_train_extra, y_train_extra = X_train_new, y_train_new
    #     print("---------------end bad epoch = {}-------------------".format(countx))
    #
    #     print("---------------start good epoch = {}-------------------".format(countx))
    #     X_train_new, y_train_new = add_ratings(X_train_new, y_train_new, user_similarity_mat, item_similarity_mat,
    #                                            helpful, X_train_extra, y_train_extra)
    #
    #     model3 = FunkSVD(num_users, num_items).to(device)
    #     loss = torch.nn.MSELoss(reduction="sum")
    #     train_ls, test_ls = train(model3, X_train_new, y_train_new, X_test, y_test, loss, num_epochs,
    #                               learning_rate, weight_decay, batch_size)
    #
    #     user_embedding_list = []
    #     item_embedding_list = []
    #     for i in range(len(model3.user_embedding.weight)):
    #         user_embedding_list.append(model3.user_embedding.weight[i].tolist())
    #     for j in range(len(model3.item_embedding.weight)):
    #         item_embedding_list.append(model3.item_embedding.weight[j].tolist())
    #     user_similarity_mat, item_similarity_mat = cal_similarity(user_embedding_list, item_embedding_list)
    #
    #     model3.eval()
    #     train_dataset = MfDataset(X_train_new[:, 0], X_train_new[:, 1], y_train_o)
    #     train_loader = DataLoader(train_dataset, shuffle=True)
    #     test_dataset = MfDataset(X_test[:, 0], X_test[:, 1], y_test)
    #     test_loader = DataLoader(test_dataset, shuffle=True)
    #     recursion_depth = 150
    #     gpu = 0
    #     r = 1
    #     influences, harmful, helpful = calc_influence_single(model3, X_train_new, y_train_new, test_loader, gpu,
    #                                                          recursion_depth, r, s_test_vec=None)
    #     X_train_extra, y_train_extra = X_train_new, y_train_new
    #     print("---------------end good epoch = {}-------------------".format(countx))
    #
    #     countx += 1


    print("\nepochs %d, mean train loss = %f, mse = %f" % (num_epochs, sum(train_ls)/len(train_ls),
                                                            sum(test_ls)/len(test_ls)))
    print("The total running time is {} minutes {} seconds"
          .format(int((time.time()-current_time)/60), (time.time()-current_time)%60))


if __name__ == '__main__':
    main()