import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pdb
import random

from tqdm import tqdm


# class VAE(nn.Module):  # 定义VAE模型
#     def __init__(self, img_size, latent_dim):  # 初始化方法
#         super(VAE, self).__init__()  # 继承初始化方法
#         self.in_channel, self.img_h, self.img_w = img_size  # 由输入图片形状得到图片通道数C、图片高度H、图片宽度W
#         self.h = self.img_h // 32  # 经过5次卷积后，最终特征层高度变为原图片高度的1/32
#         self.w = self.img_w // 32  # 经过5次卷积后，最终特征层宽度变为原图片高度的1/32
#         hw = self.h * self.w  # 最终特征层的尺寸hxw
#         self.latent_dim = latent_dim  # 采样变量Z的长度
#         self.hidden_dims = [32, 64, 128, 256, 512]  # 特征层通道数列表
#         # 开始构建编码器Encoder
#         layers = []  # 用于存放模型结构
#         for hidden_dim in self.hidden_dims:  # 循环特征层通道数列表
#             layers += [nn.Conv2d(self.in_channel, hidden_dim, 3, 2, 1),  # 添加conv
#                        nn.BatchNorm2d(hidden_dim),  # 添加bn
#                        nn.LeakyReLU()]  # 添加leakyrelu
#             self.in_channel = hidden_dim  # 将下次循环的输入通道数设为本次循环的输出通道数
#
#         self.encoder = nn.Sequential(*layers)  # 解码器Encoder模型结构
#
#         self.fc_mu = nn.Linear(self.hidden_dims[-1] * hw, self.latent_dim)  # linaer，将特征向量转化为分布均值mu
#         self.fc_var = nn.Linear(self.hidden_dims[-1] * hw, self.latent_dim)  # linear，将特征向量转化为分布方差的对数log(var)
#         # 开始构建解码器Decoder
#         layers = []  # 用于存放模型结构
#         self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1] * hw)  # linaer，将采样变量Z转化为特征向量
#         self.hidden_dims.reverse()  # 倒序特征层通道数列表
#         for i in range(len(self.hidden_dims) - 1):  # 循环特征层通道数列表
#             layers += [nn.ConvTranspose2d(self.hidden_dims[i], self.hidden_dims[i + 1], 3, 2, 1, 1),  # 添加transconv
#                        nn.BatchNorm2d(self.hidden_dims[i + 1]),  # 添加bn
#                        nn.LeakyReLU()]  # 添加leakyrelu
#         layers += [nn.ConvTranspose2d(self.hidden_dims[-1], self.hidden_dims[-1], 3, 2, 1, 1),  # 添加transconv
#                    nn.BatchNorm2d(self.hidden_dims[-1]),  # 添加bn
#                    nn.LeakyReLU(),  # 添加leakyrelu
#                    nn.Conv2d(self.hidden_dims[-1], img_size[0], 3, 1, 1),  # 添加conv
#                    nn.Tanh()]  # 添加tanh
#         self.decoder = nn.Sequential(*layers)  # 编码器Decoder模型结构
#
#     def encode(self, x):  # 定义编码过程
#         result = self.encoder(x)  # Encoder结构,(n,1,32,32)-->(n,512,1,1)
#         result = torch.flatten(result, 1)  # 将特征层转化为特征向量,(n,512,1,1)-->(n,512)
#         mu = self.fc_mu(result)  # 计算分布均值mu,(n,512)-->(n,128)
#         log_var = self.fc_var(result)  # 计算分布方差的对数log(var),(n,512)-->(n,128)
#         return [mu, log_var]  # 返回分布的均值和方差对数
#
#     def decode(self, z):  # 定义解码过程
#         y = self.decoder_input(z).view(-1, self.hidden_dims[0], self.h,
#                                        self.w)  # 将采样变量Z转化为特征向量，再转化为特征层,(n,128)-->(n,512)-->(n,512,1,1)
#         y = self.decoder(y)  # decoder结构,(n,512,1,1)-->(n,1,32,32)
#         return y  # 返回生成样本Y
#
#     def reparameterize(self, mu, log_var):  # 重参数技巧
#         std = torch.exp(0.5 * log_var)  # 分布标准差std
#         eps = torch.randn_like(std)  # 从标准正态分布中采样,(n,128)
#         return mu + eps * std  # 返回对应正态分布中的采样值
#
#     def forward(self, x):  # 前传函数
#         mu, log_var = self.encode(x)  # 经过编码过程，得到分布的均值mu和方差对数log_var
#         z = self.reparameterize(mu, log_var)  # 经过重参数技巧，得到分布采样变量Z
#         y = self.decode(z)  # 经过解码过程，得到生成样本Y
#         return [y, x, mu, log_var]  # 返回生成样本Y，输入样本X，分布均值mu，分布方差对数log_var
#
#     def sample(self, n, cuda):  # 定义生成过程
#         z = torch.randn(n, self.latent_dim)  # 从标准正态分布中采样得到n个采样变量Z，长度为latent_dim
#         if cuda:  # 如果使用cuda
#             z = z.cuda()  # 将采样变量Z加载到GPU
#         images = self.decode(z)  # 经过解码过程，得到生成样本Y
#         return images  # 返回生成样本Y
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # 开始构建编码器Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LeakyReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # 开始构建解码器Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Tanh(),
        )

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    def sample(self, n, cuda):
        z = torch.randn(n, self.latent_dim)
        if cuda:
            z = z.cuda()
        return self.decode(z)


def loss_fn(y, x, mu, log_var):  # 定义损失函数
    recons_loss = F.mse_loss(y, x)  # 重建损失，MSE
    kld_loss = 0
    #kld_loss = torch.mean(0.5 * torch.sum(mu ** 2 + torch.exp(log_var) - log_var - 1, 1), 0)  # 分布损失，正态分布与标准正态分布的KL散度
    return recons_loss + w * kld_loss  # 最终损失由两部分组成，其中分布损失需要乘上一个系数w


if __name__ == "__main__":
    total_epochs = 10000  # epochs
    batch_size = 64  # batch size
    lr = 5e-4  # lr
    w = 0.0001  # kld_loss的系数w
    num_workers = 8  # 数据加载线程数
    image_size = 32  # 图片尺寸
    image_channel = 1  # 图片通道
    latent_dim = 128  # 采样变量Z长度
    sample_images_dir = "sample_images"  # 生成样本示例存放路径
    train_dataset_dir = "../dataset/mnist"  # 训练样本存放路径

    os.makedirs(sample_images_dir, exist_ok=True)  # 创建生成样本示例存放路径
    os.makedirs(train_dataset_dir, exist_ok=True)  # 创建训练样本存放路径
    cuda = True if torch.cuda.is_available() else False  # 如果cuda可用，则使用cuda
    user_embeddings = np.load('user_embedding_list_81.npy')
    item_embeddings = np.load('item_embedding_list_81.npy')
    user_embeddings = torch.tensor(user_embeddings)
    item_embeddings = torch.tensor(item_embeddings)
    pdb.set_trace()
    VAE_dataset = []
    for i in range(3000):
        u_a = random.randint(1, len(user_embeddings)-1)
        i_b = random.randint(1, len(item_embeddings)-1)
        VAE_dataset.append(torch.cat((user_embeddings[u_a], item_embeddings[i_b]), 0))
    tensor_data = torch.stack(VAE_dataset)
    tensor_dataset = TensorDataset(tensor_data)
    batch_size = 64
    VAE_dataloader = DataLoader(dataset=tensor_dataset, batch_size=batch_size, shuffle=True)
    # img_size = (image_channel, image_size, image_size)  # 输入样本形状(1,32,32)

    vae = VAE(input_dim=128, hidden_dims=[256, 128, 32], latent_dim=64)  # 实例化VAE模型，传入输入样本形状与采样变量长度
    if cuda:  # 如果使用cuda
        vae = vae.cuda()  # 将模型加载到GPU
    # dataset and dataloader
    # transform = transforms.Compose(  # 图片预处理方法
    #     [transforms.Resize(image_size),  # 图片resize，(28x28)-->(32,32)
    #      transforms.ToTensor(),  # 转化为tensor
    #      transforms.Normalize([0.5], [0.5])]  # 标准化
    # )
    # dataloader = DataLoader(  # 定义dataloader
    #     dataset=datasets.MNIST(root=train_dataset_dir,  # 使用mnist数据集，选择数据路径
    #                            train=True,  # 使用训练集
    #                            transform=transform,  # 图片预处理
    #                            download=True),  # 自动下载
    #     batch_size=batch_size,  # batch size
    #     num_workers=num_workers,  # 数据加载线程数
    #     shuffle=True  # 打乱数据
    # )
    # optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)  # 使用Adam优化器
    # train loop
    for epoch in range(total_epochs):  # 循环epoch
        total_loss = 0  # 记录总损失
        pbar = tqdm(total=len(VAE_dataloader), desc=f"Epoch {epoch + 1}/{total_epochs}", postfix=dict,
                    miniters=0.3)  # 设置当前epoch显示进度
        for i, img in enumerate(VAE_dataloader):  # 循环iter
            img = img[0]
            img = img.to(torch.float32)
            if cuda:  # 如果使用cuda
                img = img.cuda()  # 将训练数据加载到GPU
            vae.train()  # 模型开始训练
            optimizer.zero_grad()  # 模型清零梯度
            y, x, mu, log_var = vae(img)  # 输入训练样本X，得到生成样本Y，输入样本X，分布均值mu，分布方差对数log_var
            loss = loss_fn(y, x, mu, log_var)  # 计算loss
            loss.backward()  # 反向传播，计算当前梯度
            optimizer.step()  # 根据梯度，更新网络参数
            total_loss += loss.item()  # 累计loss
            pbar.set_postfix(**{"Loss": loss.item()})  # 显示当前iter的loss
            pbar.update(1)  # 步进长度
            if epoch == total_epochs - 1:
                pdb.set_trace()
        pbar.close()  # 关闭当前epoch显示进度

        print("total_loss:%.4f" %
              (total_loss / len(VAE_dataloader)))  # 显示当前epoch训练完成后，模型的总损失
        # vae.eval()  # 模型开始验证
        # sample_images = vae.sample(25, cuda)  # 获得25个生成样本
        # save_image(sample_images.data, "%s/ep%d.png" % (sample_images_dir, (epoch + 1)), nrow=5,
        #            normalize=True)  # 保存生成样本示例(5x5)