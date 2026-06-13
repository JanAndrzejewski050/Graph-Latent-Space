from dataclasses import dataclass, field
from typing import Optional
import os
import torch


SUPPORTED_ATOMS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
NUM_ATOM_TYPES = len(SUPPORTED_ATOMS)
ATOM_TO_IDX = {atom: idx for idx, atom in enumerate(SUPPORTED_ATOMS)}

BOND_TYPES = {
    "NONE": 0,
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 4,
}
NUM_BOND_TYPES = len(BOND_TYPES)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TrainConfig:
    arch: str = "enhanced_gcn_vae"

    train_data: str = "data/subset/train.pt"
    val_data: str = "data/subset/val.pt"
    test_data: str = "data/subset/test.pt"
    max_nodes: int = 37

    # Number of input node features in the .pt files.
    # Set to 10 if using one-hot atom types only,
    # or 30 if using the rich feature set from GCN2.
    node_features: int = 10
    use_onehot_only: bool = True

    # Hiper parameters of model 
    hidden_dim: int = 256
    latent_dim: int = 256
    num_encoder_layers: int = 4
    dropout: float = 0.1

    # hiper parameters of training
    batch_size: int = 128
    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    beta_start: float = 0.0
    beta_end: float = 0.05
    beta_warmup_epochs: int = 10  
    beta_cyclical: bool = False  

    grad_clip_norm: float = 1.0  # reguralization - grad clipping 

    output_dir: str = "outputs"
    save_every: int = 5 

    eval_num_samples: int = 1000
    edge_threshold: float = 0.5  

    use_multi_bond: bool = True   

    def abs_path(self, rel: str) -> str:
        """Resolve a project-relative path to an absolute path."""
        return os.path.join(get_project_root(), rel)

    def get_beta(self, epoch: int) -> float:
        """
        Compute β for the given epoch using the annealing schedule.
        """
        if self.beta_cyclical:
            # Cyclical: ramp up over warmup epochs, then reset
            cycle_pos = epoch % self.beta_warmup_epochs
            ratio = cycle_pos / max(self.beta_warmup_epochs, 1)
        else:
            # Linear warmup then constant
            ratio = min(epoch / max(self.beta_warmup_epochs, 1), 1.0)
        return self.beta_start + (self.beta_end - self.beta_start) * ratio
