import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader
from tqdm import tqdm

class GraphVAEEvaluator:
    def __init__(self, model, device, max_nodes, train_smiles_list=None):
        self.model = model
        self.model.eval()
        self.device = device
        self.max_nodes = max_nodes
        self.train_smiles_set = set(train_smiles_list) if train_smiles_list else set()

    def _logits_to_mol(self, adj_logits, threshold=0.0):
        """
        Konwertuje wyjściową macierz modelu z powrotem na obiekt RDKit Mol.
        Model zwraca logity (przed funkcją sigmoid), więc próg to 0.0.
        """
        # Konwersja na binarną macierz (1 = wiązanie, 0 = brak)
        adj = (adj_logits > threshold).cpu().numpy()
        
        degrees = adj.sum(axis=0)
        active_nodes = np.where(degrees > 0)[0]
        
        if len(active_nodes) == 0:
            return None 
            
        mol = Chem.RWMol()
        node_to_idx = {}
        
        for node in active_nodes:
            idx = mol.AddAtom(Chem.Atom(6))
            node_to_idx[node] = idx
            
        for i in range(len(active_nodes)):
            for j in range(i + 1, len(active_nodes)):
                orig_i, orig_j = active_nodes[i], active_nodes[j]
                if adj[orig_i, orig_j] == 1:
                    mol.AddBond(node_to_idx[orig_i], node_to_idx[orig_j], Chem.rdchem.BondType.SINGLE)
                    
        try:
            Chem.SanitizeMol(mol)
            return mol.GetMol()
        except:
            return None

    def evaluate_vun(self, num_samples=1000):
        """
        Metryki: Validity, Uniqueness, Novelty (VUN)
        """
        print(f"Sampling {num_samples} random vectors from latent space...")
        valid_mols = []
        valid_smiles = []
        
        latent_dim = self.model.dec1.in_features
        
        with torch.no_grad():
            z = torch.randn(num_samples, latent_dim).to(self.device)
            adj_preds = self.model.decode(z)
            
            for i in tqdm(range(num_samples)):
                mol = self._logits_to_mol(adj_preds[i])
                if mol is not None:
                    valid_mols.append(mol)
                    valid_smiles.append(Chem.MolToSmiles(mol))
                    
        validity = len(valid_mols) / num_samples
        
        unique_smiles = set(valid_smiles)
        uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0.0
        
        novel_smiles = unique_smiles - self.train_smiles_set
        novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0.0
        
        print("\n--- VUN Analysis ---")
        print(f"Validity (Poprawność):  {validity*100:.2f}% ({len(valid_mols)}/{num_samples})")
        print(f"Uniqueness (Unikalność): {uniqueness*100:.2f}% ({len(unique_smiles)}/{len(valid_smiles)})")
        print(f"Unique smiles generated: {unique_smiles}")
        print(f"Novelty (Nowość):       {novelty*100:.2f}% ({len(novel_smiles)}/{len(unique_smiles)})")
        
        return validity, uniqueness, novelty

    def visualize_latent_space(self, dataloader):
        print("Extracting latent vectors and calculating properties...")
        z_list = []
        logp_list = []
        mol_wt_list = []
        
        with torch.no_grad():
            for data in tqdm(dataloader):
                data = data.to(self.device)
                mu, _ = self.model.encode(data.x, data.edge_index, data.batch)
                z_list.append(mu.cpu().numpy())
                
                # Obliczamy właściwości chemiczne oryginalnych SMILES
                for smiles in data.smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        logp_list.append(Descriptors.MolLogP(mol))
                        mol_wt_list.append(Descriptors.MolWt(mol))
                    else:
                        logp_list.append(0.0)
                        mol_wt_list.append(0.0)

        Z = np.vstack(z_list)
        
        print("Running t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42)
        Z_tsne = tsne.fit_transform(Z)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sc1 = axes[0].scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=logp_list, cmap='coolwarm', s=10, alpha=0.7)
        axes[0].set_title("Latent Space (t-SNE) Colored by LogP")
        plt.colorbar(sc1, ax=axes[0], label="LogP (Lipophilicity)")
        
        sc2 = axes[1].scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=mol_wt_list, cmap='viridis', s=10, alpha=0.7)
        axes[1].set_title("Latent Space (t-SNE) Colored by Molecular Weight")
        plt.colorbar(sc2, ax=axes[1], label="Molecular Weight (Da)")
        
        plt.tight_layout()
        plt.savefig('latent_space_analysis.png')
        print("Latent space visualization saved as 'latent_space_analysis.png'")
        plt.show()

if __name__ == '__main__':
    pass