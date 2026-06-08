import torch
import torch.nn as nn
class Reparameterization(nn.Module):
    def forward(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std
