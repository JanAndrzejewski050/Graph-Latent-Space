"""
Enhanced GCN-VAE:
  1. Deeper encoder: configurable number of GCN layers with residual
     connections, batch normalisation, and dropout.
  2. Multi-bond type prediction: predicts 5 bond types per edge
     (none / single / double / triple / aromatic) instead of binary adj.
  3. Symmetry-enforced decoder: generates only the upper triangle
     of the adjacency and mirrors it.
  4. Richer decoder: 3-layer MLPs with layer normalisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from src.models.base import GraphVAEBase
from src.model_registry import register
from src.config import TrainConfig, NUM_ATOM_TYPES, NUM_BOND_TYPES


class ResidualGCNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.bn(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h + self.skip(x)

@register("enhanced_gcn_vae")
class EnhancedGCNVAE(GraphVAEBase):
    def __init__(self, config: TrainConfig):
        super().__init__()

        node_in_dim = NUM_ATOM_TYPES if config.use_onehot_only else config.node_features
        hidden_dim = config.hidden_dim
        latent_dim = config.latent_dim
        max_nodes = config.max_nodes
        num_atom_types = NUM_ATOM_TYPES
        n_layers = config.num_encoder_layers
        dropout = config.dropout
        use_multi_bond = config.use_multi_bond

        self.max_nodes = max_nodes
        self.num_atom_types = num_atom_types
        self.latent_dim = latent_dim
        self.num_bond_types = NUM_BOND_TYPES if use_multi_bond else 1

        # Encoder
        encoder_layers = []
        in_dim = node_in_dim
        for i in range(n_layers):
            out_dim = hidden_dim
            encoder_layers.append(ResidualGCNBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.encoder = nn.ModuleList(encoder_layers)

        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder: bond types
        self._tri_len = max_nodes * (max_nodes - 1) // 2

        bond_out_dim = self._tri_len * self.num_bond_types

        self.dec_adj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bond_out_dim),
        )

        # Decoder: atom types
        self.dec_node = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_nodes * num_atom_types),
        )


    def encode(self, x, edge_index, batch):
        h = x
        for layer in self.encoder:
            h = layer(h, edge_index)
        h = global_mean_pool(h, batch)
        return self.mu_layer(h), self.logvar_layer(h)


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu


    def decode(self, z):
        batch_size = z.size(0)
        N = self.max_nodes

        adj_raw = self.dec_adj(z)

        if self.is_multi_bond:
            adj_raw = adj_raw.view(batch_size, self.num_bond_types, self._tri_len)
            adj_pred = self._tri_to_symmetric(adj_raw, batch_size, N)
        else:
            adj_raw = adj_raw.view(batch_size, 1, self._tri_len)
            adj_pred = self._tri_to_symmetric(adj_raw, batch_size, N)
            adj_pred = adj_pred.squeeze(1)

        node_pred = self.dec_node(z).view(batch_size, N, self.num_atom_types)

        return adj_pred, node_pred

    def _tri_to_symmetric(self, tri_vals, batch_size, N):
        """
        Convert upper-triangle values to a full symmetric matrix.
        """
        C = tri_vals.size(1)
        device = tri_vals.device

        # Build index mask for upper triangle 
        idx = torch.triu_indices(N, N, offset=1, device=device)

        out = torch.zeros(batch_size, C, N, N, device=device)
        out[:, :, idx[0], idx[1]] = tri_vals
        out = out + out.transpose(-2, -1)  # mirror to lower triangle
        return out
