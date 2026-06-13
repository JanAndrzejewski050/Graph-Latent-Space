"""
Graph VAE model implementations.

All models conform to the GraphVAEBase interface defined in base.py.
Use model_registry to instantiate models by name.
"""

from src.models.base import GraphVAEBase
from src.models.gcn_vae_v1 import GCNVAEv1
from src.models.enhanced_gcn_vae import EnhancedGCNVAE

__all__ = [
    "GraphVAEBase",
    "GCNVAEv1",
    "EnhancedGCNVAE",
]
