import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from src.models.base import GraphVAEBase
from src.model_registry import register
from src.config import TrainConfig, NUM_ATOM_TYPES


@register("gcn_vae_v1")
class GCNVAEv1(GraphVAEBase):
    """
    Original 2-layer GCN VAE with separate MLP decoders for
    adjacency (binary) and atom types.
    """

    def __init__(self, config: TrainConfig):
        super().__init__()

        node_in_dim = NUM_ATOM_TYPES if config.use_onehot_only else config.node_features
        hidden_dim = config.hidden_dim
        latent_dim = config.latent_dim
        max_nodes = config.max_nodes
        num_atom_types = NUM_ATOM_TYPES

        self.max_nodes = max_nodes
        self.num_atom_types = num_atom_types
        self.latent_dim = latent_dim
        self.num_bond_types = 1
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.dec_adj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * max_nodes),
        )

        self.dec_node = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * num_atom_types),
        )

    def encode(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = global_mean_pool(h, batch)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        # Adjacency
        adj_out = self.dec_adj(z)
        adj_pred = adj_out.view(-1, self.max_nodes, self.max_nodes)
        # Symmetrise
        adj_pred = (adj_pred + adj_pred.transpose(1, 2)) / 2.0

        # Node types
        node_out = self.dec_node(z)
        node_pred = node_out.view(-1, self.max_nodes, self.num_atom_types)

        return adj_pred, node_pred
