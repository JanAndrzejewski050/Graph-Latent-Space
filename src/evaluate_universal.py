import json
import os
import sys

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors, QED as QEDModule
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.config import (
    SUPPORTED_ATOMS,
    NUM_ATOM_TYPES,
    BOND_TYPES,
    NUM_BOND_TYPES,
    TrainConfig,
)

_sascorer = None

def _get_sascorer():
    global _sascorer
    if _sascorer is not None:
        return _sascorer
    try:
        notebook_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "graphVAE", "notebooks"
        )
        if notebook_dir not in sys.path:
            sys.path.insert(0, notebook_dir)
        import sascorer
        _sascorer = sascorer
        return _sascorer
    except ImportError:
        print("WARNING: sascorer not found. SA-score evaluation disabled.")
        return None


RDKIT_BOND_TYPES = {
    BOND_TYPES["SINGLE"]:   Chem.rdchem.BondType.SINGLE,
    BOND_TYPES["DOUBLE"]:   Chem.rdchem.BondType.DOUBLE,
    BOND_TYPES["TRIPLE"]:   Chem.rdchem.BondType.TRIPLE,
    BOND_TYPES["AROMATIC"]: Chem.rdchem.BondType.AROMATIC,
}


class UniversalEvaluator:
    """
    Universal evaluator for any GraphVAEBase model.

    Usage:
        evaluator = UniversalEvaluator(model, device)
        validity, uniqueness, novelty, valid_mols = evaluator.evaluate_vun(1000)
        evaluator.evaluate_properties(valid_mols)
        evaluator.visualize_latent_space(dataloader)
    """

    def __init__(self, model, device, train_smiles_list=None):
        self.model = model
        self.model.eval()
        self.device = device
        self.train_smiles_set = set(train_smiles_list) if train_smiles_list else set()

    # molecule decoder
    def logits_to_mol(self, adj_logits, node_logits, threshold=0.5):
        """
        Convert model outputs to an RDKit Mol object.
        Supports both binary adjacency [N, N] and multi-bond [C, N, N].
        """
        is_multi = adj_logits.dim() == 3  

        # Atom types
        atom_type_indices = torch.argmax(node_logits, dim=-1).cpu().numpy()

        # Adjacency / bond types
        if is_multi:
            bond_type_matrix = torch.argmax(adj_logits, dim=0).cpu().numpy() 
        else:
            adj = (adj_logits > threshold).cpu().numpy()
            bond_type_matrix = adj.astype(np.int64)

        N = bond_type_matrix.shape[0]

        has_bond = np.zeros(N, dtype=bool)
        for i in range(N):
            for j in range(N):
                if bond_type_matrix[i, j] != BOND_TYPES["NONE"]:
                    has_bond[i] = True
                    has_bond[j] = True
        active_nodes = np.where(has_bond)[0]

        if len(active_nodes) == 0:
            return None

        mol = Chem.RWMol()
        node_to_idx = {}

        for node in active_nodes:
            pred_idx = atom_type_indices[node]
            if pred_idx >= len(SUPPORTED_ATOMS):
                pred_idx = 1  # default to Carbon
            atomic_num = SUPPORTED_ATOMS[pred_idx]
            idx = mol.AddAtom(Chem.Atom(atomic_num))
            node_to_idx[node] = idx

        for i_pos in range(len(active_nodes)):
            for j_pos in range(i_pos + 1, len(active_nodes)):
                orig_i, orig_j = active_nodes[i_pos], active_nodes[j_pos]
                bond_type_idx = bond_type_matrix[orig_i, orig_j]

                if bond_type_idx != BOND_TYPES["NONE"]:
                    rdkit_bond = RDKIT_BOND_TYPES.get(
                        bond_type_idx, Chem.rdchem.BondType.SINGLE
                    )
                    try:
                        mol.AddBond(node_to_idx[orig_i], node_to_idx[orig_j], rdkit_bond)
                    except Exception:
                        pass

        try:
            Chem.SanitizeMol(mol)
            return mol.GetMol()
        except Exception:
            return None

  
    def evaluate_vun(self, num_samples=1000, threshold=0.5):
        print(f"Sampling {num_samples} molecules from latent space...")
        valid_mols = []
        valid_smiles = []

        with torch.no_grad():
            adj_preds, node_preds = self.model.sample(num_samples, self.device)

            for i in tqdm(range(num_samples), desc="Decoding"):
                mol = self.logits_to_mol(adj_preds[i], node_preds[i], threshold)
                if mol is not None:
                    valid_mols.append(mol)
                    valid_smiles.append(Chem.MolToSmiles(mol))

        validity = len(valid_mols) / num_samples
        unique_smiles = set(valid_smiles)
        uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0.0
        novel_smiles = unique_smiles - self.train_smiles_set
        novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0.0

        print("\n╔══════════════════════════════════════╗")
        print("║         VUN Analysis Results         ║")
        print("╠══════════════════════════════════════╣")
        print(f"║  Validity:   {validity*100:6.2f}% ({len(valid_mols):>5}/{num_samples:<5}) ║")
        print(f"║  Uniqueness: {uniqueness*100:6.2f}% ({len(unique_smiles):>5}/{len(valid_smiles):<5}) ║")
        print(f"║  Novelty:    {novelty*100:6.2f}% ({len(novel_smiles):>5}/{len(unique_smiles):<5}) ║")
        print("╚══════════════════════════════════════╝")

        return validity, uniqueness, novelty, valid_mols, valid_smiles

    def compute_properties(self, mols):
        """Compute molecular properties for a list of RDKit Mol objects."""
        sascorer = _get_sascorer()
        records = []
        for mol in tqdm(mols, desc="Computing properties"):
            if mol is None:
                continue
            try:
                rec = {
                    "molWt": Descriptors.MolWt(mol),
                    "HeavyAtomCount": mol.GetNumHeavyAtoms(),
                    "cLogP": Descriptors.MolLogP(mol),
                    "TPSA": Descriptors.TPSA(mol),
                    "HBD": rdMolDescriptors.CalcNumHBD(mol),
                    "HBA": rdMolDescriptors.CalcNumHBA(mol),
                    "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                    "RingCount": rdMolDescriptors.CalcNumRings(mol),
                    "AromaticRingCount": rdMolDescriptors.CalcNumAromaticRings(mol),
                    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
                    "QED": QEDModule.qed(mol),
                }
                if sascorer is not None:
                    rec["SA_score"] = sascorer.calculateScore(mol)
                records.append(rec)
            except Exception:
                continue
        return pd.DataFrame(records)

    def plot_property_comparison(self, gen_props, ref_props, save_path=None):
        """
        Side-by-side histograms comparing generated vs. reference properties.
        """
        properties = [c for c in gen_props.columns if c in ref_props.columns]
        n = len(properties)
        cols = 3
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = axes.flatten() if n > 1 else [axes]

        for idx, prop in enumerate(properties):
            ax = axes[idx]
            ax.hist(ref_props[prop].dropna(), bins=50, alpha=0.5, label="Training data",
                    density=True, color="#2196F3")
            ax.hist(gen_props[prop].dropna(), bins=50, alpha=0.5, label="Generated",
                    density=True, color="#FF5722")
            ax.set_title(prop, fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Property Distribution: Generated vs. Training Data",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_sa_scores(self, gen_mols, ref_mols=None, save_path=None):
        """Plot SA-score distribution for generated (and optionally reference) molecules."""
        sascorer = _get_sascorer()
        if sascorer is None:
            print("SA-scorer not available; skipping.")
            return

        gen_scores = []
        for mol in tqdm(gen_mols, desc="SA-scores (generated)"):
            try:
                gen_scores.append(sascorer.calculateScore(mol))
            except Exception:
                continue

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(gen_scores, bins=50, alpha=0.6, label="Generated", density=True, color="#FF5722")

        if ref_mols:
            ref_scores = []
            for mol in tqdm(ref_mols, desc="SA-scores (reference)"):
                try:
                    ref_scores.append(sascorer.calculateScore(mol))
                except Exception:
                    continue
            ax.hist(ref_scores, bins=50, alpha=0.6, label="Training data", density=True, color="#2196F3")

        ax.set_xlabel("SA Score (lower = easier to synthesise)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Synthetic Accessibility Score Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"Generated SA-score:  mean={np.mean(gen_scores):.2f}  std={np.std(gen_scores):.2f}")
        if ref_mols:
            print(f"Reference SA-score:  mean={np.mean(ref_scores):.2f}  std={np.std(ref_scores):.2f}")

    
    def visualize_latent_space(self, dataloader, config=None, max_points=5000,
                               save_path=None):
        z_list = []
        logp_list = []
        mol_wt_list = []
        count = 0

        with torch.no_grad():
            for data in tqdm(dataloader, desc="Encoding"):
                data = data.to(self.device)
                if config and config.use_onehot_only and data.x.size(1) > NUM_ATOM_TYPES:
                    data.x = data.x[:, :NUM_ATOM_TYPES]

                mu, _ = self.model.encode(data.x, data.edge_index, data.batch)
                z_list.append(mu.cpu().numpy())

                # Try to get molecular properties from SMILES
                if hasattr(data, 'smiles'):
                    for smiles in data.smiles:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            logp_list.append(Descriptors.MolLogP(mol))
                            mol_wt_list.append(Descriptors.MolWt(mol))
                        else:
                            logp_list.append(0.0)
                            mol_wt_list.append(0.0)
                else:
                    # Fill with zeros if no SMILES
                    n = mu.size(0)
                    logp_list.extend([0.0] * n)
                    mol_wt_list.extend([0.0] * n)

                count += mu.size(0)
                if count >= max_points:
                    break

        Z = np.vstack(z_list)[:max_points]
        logp_list = logp_list[:max_points]
        mol_wt_list = mol_wt_list[:max_points]

        print(f"Running t-SNE on {Z.shape[0]} points...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, Z.shape[0] - 1))
        Z_tsne = tsne.fit_transform(Z)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sc1 = axes[0].scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=logp_list,
                              cmap='coolwarm', s=10, alpha=0.7)
        axes[0].set_title("Latent Space (t-SNE) — LogP", fontsize=13, fontweight="bold")
        plt.colorbar(sc1, ax=axes[0], label="LogP")

        sc2 = axes[1].scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=mol_wt_list,
                              cmap='viridis', s=10, alpha=0.7)
        axes[1].set_title("Latent Space (t-SNE) — Molecular Weight", fontsize=13, fontweight="bold")
        plt.colorbar(sc2, ax=axes[1], label="Mol Wt (Da)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

  
    def draw_molecule_gallery(self, mols, n=20, mols_per_row=5, save_path=None):
        """Draw a grid of generated molecules."""
        mols_to_draw = [m for m in mols if m is not None][:n]
        legends = [Chem.MolToSmiles(m) for m in mols_to_draw]

        # Truncate long SMILES for display
        legends = [s[:30] + "…" if len(s) > 30 else s for s in legends]

        img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=mols_per_row,
                                   legends=legends, subImgSize=(300, 300))
        if save_path:
            img.save(save_path)
        return img


    @staticmethod
    def plot_training_history(history_path, save_path=None):
        """
        Plot training curves from a saved JSON history file.
        """
        with open(history_path, "r") as f:
            history = json.load(f)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        epochs = range(1, len(history["train_loss"]) + 1)

        # Total loss
        axes[0, 0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
        axes[0, 0].plot(epochs, history["val_loss"], label="Val", linewidth=2)
        axes[0, 0].set_title("Total Loss", fontsize=13, fontweight="bold")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Adjacency loss
        axes[0, 1].plot(epochs, history["train_adj_loss"], label="Train", linewidth=2)
        axes[0, 1].plot(epochs, history["val_adj_loss"], label="Val", linewidth=2)
        axes[0, 1].set_title("Adjacency Reconstruction Loss", fontsize=13, fontweight="bold")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Node loss
        axes[1, 0].plot(epochs, history["train_node_loss"], label="Train", linewidth=2)
        axes[1, 0].plot(epochs, history["val_node_loss"], label="Val", linewidth=2)
        axes[1, 0].set_title("Node Reconstruction Loss", fontsize=13, fontweight="bold")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # KL divergence + beta
        ax_kl = axes[1, 1]
        ax_kl.plot(epochs, history["train_kl_loss"], label="KL (train)", linewidth=2)
        ax_kl.set_title("KL Divergence & β Schedule", fontsize=13, fontweight="bold")
        ax_kl.set_ylabel("KL Divergence")
        ax_kl.grid(alpha=0.3)

        if "beta" in history:
            ax_beta = ax_kl.twinx()
            ax_beta.plot(epochs, history["beta"], label="β", linewidth=2,
                        color="red", linestyle="--")
            ax_beta.set_ylabel("β", color="red")
            ax_beta.tick_params(axis="y", labelcolor="red")

        ax_kl.legend(loc="upper left")

        plt.suptitle("Training History", fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
