import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt
import os

TRAIN_DATA_PATH = '../data/subset/train.pt'
VAL_DATA_PATH = '../data/subset/val.pt'

NODE_FEATURES = 1 
HIDDEN_DIM = 64
LATENT_DIM = 128
MAX_NODES = 37
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 1e-3
BETA = 0.001

SUPPORTED_ATOMS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
NUM_ATOM_TYPES = len(SUPPORTED_ATOMS)
NODE_FEATURES = NUM_ATOM_TYPES # one hot vector

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"using device: {DEVICE}")

def create_one_hot_features(x, supported_atoms, device):
    atom_to_idx = {atom: idx for idx, atom in enumerate(supported_atoms)}
    indices = []
    for atomic_num in x.view(-1).tolist():
        indices.append(atom_to_idx.get(int(atomic_num), 1)) # Default is Carbon
        
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    return F.one_hot(indices_tensor, num_classes=len(supported_atoms)).float()

class GraphVAE(nn.Module):
    def __init__(self, node_in_dim, hidden_dim, latent_dim, max_nodes, num_atom_types):
        super(GraphVAE, self).__init__()
        self.max_nodes = max_nodes
        self.num_atom_types = num_atom_types
        
        # encoder
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # decoder 1 - bonds
        self.dec_adj1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_adj2 = nn.Linear(hidden_dim, max_nodes * max_nodes) 
        
        # decoder 2 - atom types
        self.dec_node1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_node2 = nn.Linear(hidden_dim, max_nodes * num_atom_types)
        
    def encode(self, x, edge_index, batch):
        x_one_hot = create_one_hot_features(x, SUPPORTED_ATOMS, x.device)
        
        x = F.relu(self.conv1(x_one_hot, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch) 
        return self.mu_layer(x), self.logvar_layer(x)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
        
    def decode(self, z):
        # decoding bonds
        h_adj = F.relu(self.dec_adj1(z))
        adj_out = self.dec_adj2(h_adj)
        adj_pred = adj_out.view(-1, self.max_nodes, self.max_nodes)
        
        # decoding atoms
        h_node = F.relu(self.dec_node1(z))
        node_out = self.dec_node2(h_node)
        node_pred = node_out.view(-1, self.max_nodes, self.num_atom_types)
        
        return adj_pred, node_pred
        
    def forward(self, data):
        mu, logvar = self.encode(data.x, data.edge_index, data.batch)
        z = self.reparameterize(mu, logvar)
        adj_pred, node_pred = self.decode(z)
        return adj_pred, node_pred, mu, logvar


def loss_function(adj_pred, adj_target, node_pred, node_target, node_mask, mu, logvar, beta):
    # loss of bonds reconstruction
    recon_adj_loss = F.binary_cross_entropy_with_logits(adj_pred, adj_target, reduction='mean')    
    
    # loss of atoms reconstruction
    node_pred_flat = node_pred.view(-1, NUM_ATOM_TYPES)
    node_target_flat = node_target.view(-1)
    mask_flat = node_mask.view(-1).bool()
    
    if mask_flat.sum() > 0:
        recon_node_loss = F.cross_entropy(node_pred_flat[mask_flat], node_target_flat[mask_flat], reduction='mean')
    else:
        recon_node_loss = 0.0
        
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_adj_loss + recon_node_loss + beta * kl_loss
    return total_loss, recon_adj_loss, recon_node_loss, kl_loss


def train(model, loader, optimizer):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0
    
    atom_to_idx = {atom: idx for idx, atom in enumerate(SUPPORTED_ATOMS)}
    
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        
        adj_target = to_dense_adj(data.edge_index, data.batch, max_num_nodes=MAX_NODES)
        if adj_target.size(1) < MAX_NODES:
            pad_size = MAX_NODES - adj_target.size(1)
            adj_target = F.pad(adj_target, (0, pad_size, 0, pad_size))
        adj_target = adj_target.to(DEVICE)
        
        batch_size = data.batch.max().item() + 1
        node_target = torch.zeros(batch_size, MAX_NODES, dtype=torch.long, device=DEVICE)
        node_mask = torch.zeros(batch_size, MAX_NODES, dtype=torch.bool, device=DEVICE)
        
        current_idx = 0
        for i in range(batch_size):
            num_nodes = (data.batch == i).sum().item() 
            num_nodes = min(num_nodes, MAX_NODES)
            
            for j in range(num_nodes):
                atomic_num = int(data.x[current_idx + j][0].item())
                node_target[i, j] = atom_to_idx.get(atomic_num, 1) 
                node_mask[i, j] = True
                
            current_idx += (data.batch == i).sum().item()
            
        adj_pred, node_pred, mu, logvar = model(data)
        
        loss, recon_adj, recon_node, kl = loss_function(
            adj_pred, adj_target, node_pred, node_target, node_mask, mu, logvar, BETA
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        total_recon += (recon_adj.item() + recon_node.item()) * data.num_graphs 
        total_kl += kl.item() * data.num_graphs
        
    return total_loss / len(loader.dataset), total_recon / len(loader.dataset), total_kl / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss, total_recon, total_kl = 0, 0, 0
    
    # Słownik do mapowania: Liczba Atomowa -> Indeks (tak samo jak w pętli train)
    atom_to_idx = {atom: idx for idx, atom in enumerate(SUPPORTED_ATOMS)}
    
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            
            adj_target = to_dense_adj(data.edge_index, data.batch, max_num_nodes=MAX_NODES)
            if adj_target.size(1) < MAX_NODES:
                pad_size = MAX_NODES - adj_target.size(1)
                adj_target = F.pad(adj_target, (0, pad_size, 0, pad_size))
            adj_target = adj_target.to(DEVICE)
            
            batch_size = data.batch.max().item() + 1
            node_target = torch.zeros(batch_size, MAX_NODES, dtype=torch.long, device=DEVICE)
            node_mask = torch.zeros(batch_size, MAX_NODES, dtype=torch.bool, device=DEVICE)
            
            current_idx = 0
            for i in range(batch_size):
                num_nodes = (data.batch == i).sum().item() 
                num_nodes = min(num_nodes, MAX_NODES)
                
                for j in range(num_nodes):
                    atomic_num = int(data.x[current_idx + j][0].item())
                    node_target[i, j] = atom_to_idx.get(atomic_num, 1) 
                    node_mask[i, j] = True
                    
                current_idx += (data.batch == i).sum().item()
                
            adj_pred, node_pred, mu, logvar = model(data)
            
            loss, recon_adj, recon_node, kl = loss_function(
                adj_pred, adj_target, node_pred, node_target, node_mask, mu, logvar, BETA
            )
            
            total_loss += loss.item() * data.num_graphs
            total_recon += (recon_adj.item() + recon_node.item()) * data.num_graphs
            total_kl += kl.item() * data.num_graphs
            
    return total_loss / len(loader.dataset), total_recon / len(loader.dataset), total_kl / len(loader.dataset)

def main():
    print("loading data..")
    train_dataset = torch.load(TRAIN_DATA_PATH, weights_only=False)
    val_dataset = torch.load(VAL_DATA_PATH, weights_only=False)
    
    train_dataset = [g for g in train_dataset if g.num_nodes <= MAX_NODES]
    val_dataset = [g for g in val_dataset if g.num_nodes <= MAX_NODES]
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"training on {len(train_dataset)} graphs...")
    
    model = GraphVAE(NODE_FEATURES, HIDDEN_DIM, LATENT_DIM, MAX_NODES, NUM_ATOM_TYPES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf') 
    
    for epoch in range(1, EPOCHS + 1):
        t_loss, t_recon, t_kl = train(model, train_loader, optimizer)
        v_loss, v_recon, v_kl = evaluate(model, val_loader)
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{EPOCHS:03d} | "
                  f"Train Loss: {t_loss:.4f} (Recon: {t_recon:.4f}, KL: {t_kl:.4f}) | "
                  f"Val Loss: {v_loss:.4f}")
            
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), 'model1_weights.pth')

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title('Krzywa uczenia Graph VAE')
    plt.xlabel('Epoka')
    plt.ylabel('Loss (ELBO)')
    plt.legend()
    plt.grid(True)    
    plt.savefig('loss_curve1.png')
    plt.show()

if __name__ == '__main__':
    main()