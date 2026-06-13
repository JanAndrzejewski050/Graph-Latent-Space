import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class GraphVAEBase(nn.Module, ABC):

    def __init__(self):
        super().__init__()
        self.max_nodes: int = 0
        self.num_atom_types: int = 0
        self.latent_dim: int = 0
        self.num_bond_types: int = 1 

    @abstractmethod
    def encode(self, x, edge_index, batch):
        ...

    @abstractmethod
    def decode(self, z):
        ...

    @abstractmethod
    def reparameterize(self, mu, logvar):
        ...

    def forward(self, data):
        mu, logvar = self.encode(data.x, data.edge_index, data.batch)
        z = self.reparameterize(mu, logvar)
        adj_pred, node_pred = self.decode(z)
        return adj_pred, node_pred, mu, logvar

    
    def sample(self, num_samples, device=None):
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    @property
    def is_multi_bond(self):
        return self.num_bond_types > 1
