import torch
from torch import nn
from model.mlp import MLP

class Autoencoder(nn.Module):
    """Some Information about Autoencoder"""
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = MLP()
        self.decoder = MLP()

    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded