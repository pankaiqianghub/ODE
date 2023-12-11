import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import os
import random


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
            *modules,
            nn.Linear(hidden_dims[-1], latent_dim),
            nn.Tanh()
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


def loss_fn(y, x, mu, log_var, w):
    recons_loss = F.mse_loss(y, x)
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recons_loss + w * kld_loss


def main():
    # Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_epochs = 10000
    lr = 5e-4
    w = 0.0001
    latent_dim = 128

    # Load Data
    user_embeddings = np.load('user_embedding_list_81.npy')
    item_embeddings = np.load('item_embedding_list_81.npy')
    user_embeddings = torch.tensor(user_embeddings, dtype=torch.float32)
    item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32)
    VAE_dataset = [torch.cat((user_embeddings[random.randint(0, len(user_embeddings)-1)],
                              item_embeddings[random.randint(0, len(item_embeddings)-1)]))
                   for _ in range(3000)]
    tensor_data = torch.stack(VAE_dataset)
    tensor_dataset = TensorDataset(tensor_data)
    VAE_dataloader = DataLoader(dataset=tensor_dataset, batch_size=64, shuffle=True)

    # Model and Optimizer
    vae = VAE(input_dim=128, hidden_dims=[256, 128, 32], latent_dim=64).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # Training Loop
    for epoch in tqdm(range(total_epochs), desc="Training Epochs"):
        vae.train()
        epoch_loss = 0.0
        for i, (img,) in enumerate(VAE_dataloader):
            img = img.to(device)
            optimizer.zero_grad()
            reconstructed, mu, log_var = vae(img)
            loss = loss_fn(reconstructed, img, mu, log_var, w)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{total_epochs} - Loss: {epoch_loss / len(VAE_dataloader)}")


if __name__ == "__main__":
    main()