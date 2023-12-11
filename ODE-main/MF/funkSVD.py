import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import time
import datetime
from torch.utils.data import DataLoader, Dataset, TensorDataset
import argparse
import random
from scipy.sparse import dok_matrix
import pdb
import torch.nn.functional as F
from tqdm import tqdm


torch.random.manual_seed(22)


def parse_args():
    parser = argparse.ArgumentParser(description="Command line arguments")

    parser.add_argument('--regularization', type=float, default=1e-12,
                        help="Set regularization value")
    parser.add_argument('--batch_size', type=int, default=1024,
                        help="Set batch size value")
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help="Set device (cuda or cpu)")
    parser.add_argument('--num_epochs', type=int, default=75,
                        help="Set number of epochs")
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help="Set learning rate value")
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help="Set weight decay value")
    parser.add_argument('--dataset', type=str, default='ml-100k',
                        choices=['ml-100k', 'ml-1m'], help="Set device (ml-100k or ml-1m)")
    parser.add_argument('--recursion_depth', type=int, default=500,
                        help="Calculate Influences recursion")
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help="User and Item Embedding size")
    parser.add_argument('--VAE_epochs', type=int, default=5000,
                        help="VAE training epochs")
    parser.add_argument('--pacify_epochs', type=int, default=50,
                        help="pacify training epochs")
    parser.add_argument('--VAE_lr', type=float, default=5e-4,
                        help="VAE learning rate")
    parser.add_argument('--VAE_w', type=float, default=0.0001,
                        help="VAE weight")
    parser.add_argument('--VAE_w2', type=float, default=0.0003,
                        help="VAE weight 2")
    parser.add_argument('--target_item_num', type=int, default=5,
                        help="target_item numbers")


    args = parser.parse_args()
    return args


args = parse_args()
regularization = args.regularization
batch_size = args.batch_size
device = torch.device(args.device)
num_epochs = args.num_epochs
learning_rate = args.learning_rate
weight_decay = args.weight_decay
dataset_name = args.dataset
recursion_depth = args.recursion_depth
embedding_dim = args.embedding_dim
training_file = '../dataset/{}/{}.train.rating'.format(dataset_name, dataset_name)
testing_file = '../dataset/{}/{}.test.rating'.format(dataset_name, dataset_name)
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H")
influences_rank_npy = 'influences_rank/influences_rank_{}_{}.npy'.format(dataset_name, current_datetime)
user_embedding_npy = 'user_embedding/user_embedding_{}_{}.npy'.format(dataset_name, current_datetime)
item_embedding_npy = 'item_embedding/item_embedding_{}_{}.npy'.format(dataset_name, current_datetime)

VAE_epochs = args.VAE_epochs
pacify_epochs = args.pacify_epochs
VAE_lr = args.VAE_lr
VAE_w = args.VAE_w
VAE_w2 = args.VAE_w2
inf1 = 0
inf2 = 0
# target items
a = [[1485, 1320, 821, 1562, 1531],
     [1018, 946, 597, 575, 516],
     [1032, 1033, 797, 60, 1366],
     [1576, 926, 942, 848, 107],
     [539, 117, 1600, 1326, 208]]

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
    def __init__(self, num_users, num_items, embedding_dim, regularization):
        super(FunkSVD, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.regularization = regularization

        self.user_embedding.weight.data.uniform_(0, 0.005)
        self.item_embedding.weight.data.uniform_(0, 0.005)

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        dot_product = torch.sum(torch.mul(user_embeds, item_embeds), dim=1)
        preds = dot_product
        regularization_term = self.regularization * (torch.norm(user_embeds) + torch.norm(item_embeds))
        preds += regularization_term
        return preds


def evaluate(model, X_valid, y_valid, loss_func):
    model.eval()
    with torch.no_grad():
        n = y_valid.shape[0]
        valid_loss = loss_func(model(X_valid[:, 0], X_valid[:, 1]), y_valid)
    return valid_loss.item() / n


def train(model, X_train, y_train, X_valid, y_valid, loss_func, num_epochs, learning_rate, weight_decay, batch_size,
          X_test, y_test, num_users, num_items, patience=5):
    train_ls, valid_ls = [], []
    train_dataset = MfDataset(X_train[:, 0], X_train[:, 1], y_train)
    train_iter = DataLoader(train_dataset, batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model = model.float()

    no_improve_epochs = 0
    best_valid_loss = float('inf')

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

        current_valid_loss = evaluate(model, X_valid, y_valid, loss_func)
        valid_ls.append(current_valid_loss)

        # Early Stopping
        if current_valid_loss < best_valid_loss:
            best_valid_loss = current_valid_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        print('epoch %d, train mse %f, valid mse %f' % (epoch + 1, train_ls[-1], valid_ls[-1]))
        prediction = get_y_prediction(model, num_users, num_items)
        label = get_y_label(X_test, y_test, num_users, num_items)
        recommend_metrics(prediction, label, 10)

        if no_improve_epochs >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    return train_ls, valid_ls


def calc_influence_single(model, X_train, y_train, test_loader, gpu, recursion_depth, r, s_test_vec=None):
    # Calculate s_test vectors if not provided
    if not s_test_vec:
        x_u_test, x_i_test, y_test = test_loader.dataset[:]
        s_test_vec = calc_s_test_single(model, x_u_test, x_i_test, y_test, X_train, y_train, gpu,
                                        recursion_depth=recursion_depth, r=r)

    # Initialize train dataset and loader
    train_dataset = MfDataset(X_train[:, 0], X_train[:, 1], y_train)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

    influences = []

    # Calculate the influence for each training data
    for (x_u, x_i, y) in train_loader:
        grad_z_vec = grad_z(x_u, x_i, y, model, gpu=gpu)
        tmp_influence = -sum([torch.sum(k * j).item() for k, j in zip(grad_z_vec, s_test_vec)]) / len(train_dataset)
        influences.extend([float(tmp_influence) for _ in range(len(x_u))])

    harmful = np.argsort(influences)
    np.save('influences_emb.npy', influences)
    np.save(influences_rank_npy, harmful)

    return influences, harmful, harmful[::-1]


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
    harmful = np.load(influences_rank_npy)
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


def load_data(train_path, test_path, device):
    train_data = pd.read_csv(train_path, header=None, delimiter='\t')
    test_data = pd.read_csv(test_path, header=None, delimiter='\t')

    X_train, y_train = train_data.iloc[:, :2], train_data.iloc[:, 2]
    X_test, y_test = test_data.iloc[:, :2], test_data.iloc[:, 2]

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.int64).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.int64).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def train_model(num_users, num_items, device): #其他参数
    model = FunkSVD(num_users, num_items, embedding_dim, regularization).to(device)
    # Remaining training code...
    return model


def save_embeddings(model, user_file, item_file):
    user_embedding_list = [emb.tolist() for emb in model.user_embedding.weight]
    item_embedding_list = [emb.tolist() for emb in model.item_embedding.weight]
    np.save(user_file, user_embedding_list)
    np.save(item_file, item_embedding_list)
    return user_embedding_list, item_embedding_list


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, h_dim),
                    nn.LeakyReLU()
                )
            )
            input_dim = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[2]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def loss_fn(y, x, mu, log_var, w, w2, influence1, influence2, pos):
    recons_loss = F.mse_loss(y, x)
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    if pos == 1:
        inf_loss = influence1 - influence2
    else:
        inf_loss = influence2 - influence1
    return recons_loss + w * kld_loss+ w2 * inf_loss
def main():
    current_time = time.time()

    X_train, y_train, X_test, y_test = load_data(training_file, testing_file, device)

    num_users, num_items = X_train[:, 0].max() + 1, X_train[:, 1].max() + 1

    model = train_model(num_users, num_items, device) #其他参数
    loss = torch.nn.MSELoss(reduction="sum")
    train_ls, test_ls = train(model, X_train, y_train, X_test, y_test, loss, num_epochs,
                              learning_rate, weight_decay, batch_size, X_test, y_test, num_users, num_items)

    user_embeddings, item_embeddings = save_embeddings(model, user_embedding_npy, item_embedding_npy)

    train_dataset = MfDataset(X_train[:, 0], X_train[:, 1], y_train)
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_dataset = MfDataset(X_test[:, 0], X_test[:, 1], y_test)
    test_loader = DataLoader(test_dataset, shuffle=True)
    gpu = 0
    r = 1
    influences, harmful, helpful = calc_influence_single(model, X_train, y_train, test_loader,
                                                         gpu, recursion_depth, r, s_test_vec=None)

    print("\nepochs %d, mean train loss = %f, mse = %f" %
          (num_epochs, sum(train_ls)/len(train_ls), sum(test_ls)/len(test_ls)))
    print("The total running time is {} minutes {} seconds"
          .format(int((time.time()-current_time)/60), (time.time()-current_time)%60))

    user_embeddings = torch.tensor(user_embeddings)
    item_embeddings = torch.tensor(item_embeddings)
    VAE_dataset = [torch.cat((user_embeddings[random.randint(0, len(user_embeddings) - 1)],
                              item_embeddings[random.randint(0, len(item_embeddings) - 1)]))
                   for _ in range(3000)]
    tensor_data = torch.stack(VAE_dataset)
    tensor_dataset = TensorDataset(tensor_data)
    VAE_dataloader = DataLoader(dataset=tensor_dataset, batch_size=64, shuffle=True)

    # Model and Optimizer
    vae = VAE(input_dim=128, hidden_dims=[512, 256, 128], latent_dim=64).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=VAE_lr)

    # Training Loop
    for epoch in tqdm(range(VAE_epochs), desc="Training Epochs"):
        vae.train()
        epoch_loss = 0.0
        for i, (img,) in enumerate(VAE_dataloader):
            img = img.to(device)
            optimizer.zero_grad()
            reconstructed, mu, log_var = vae(img)
            loss = loss_fn(reconstructed, img, mu, log_var, VAE_w, VAE_w2, inf1, inf2, pos=1)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        for i, (img,) in enumerate(VAE_dataloader):
            img = img.to(device)
            optimizer.zero_grad()
            reconstructed, mu, log_var = vae(img)
            loss = loss_fn(reconstructed, img, mu, log_var, VAE_w, VAE_w2, inf1, inf2, pos=0)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{VAE_epochs} - Loss: {epoch_loss / len(VAE_dataloader)}")


if __name__ == '__main__':
    main()