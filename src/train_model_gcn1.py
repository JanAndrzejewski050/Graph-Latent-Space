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

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"using device: {DEVICE}")

class GraphVAE(nn.Module):
    def __init__(self, node_in_dim, hidden_dim, latent_dim, max_nodes):
        super(GraphVAE, self).__init__()
        self.max_nodes = max_nodes
        
        # encoder
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # latent layer
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # decoder
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, max_nodes * max_nodes) 
        
    def encode(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
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
        h = F.relu(self.dec1(z))
        out = self.dec2(h)
        return out.view(-1, self.max_nodes, self.max_nodes)
        
    def forward(self, data):
        mu, logvar = self.encode(data.x, data.edge_index, data.batch)
        z = self.reparameterize(mu, logvar)
        adj_pred = self.decode(z)
        return adj_pred, mu, logvar


def loss_function(adj_pred, adj_target, mu, logvar, beta):
    recon_loss = F.binary_cross_entropy_with_logits(adj_pred, adj_target, reduction='mean')    
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train(model, loader, optimizer):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0
    
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        
        adj_target = to_dense_adj(data.edge_index, data.batch, max_num_nodes=MAX_NODES)
        
        # matching to max_nodes 
        if adj_target.size(1) < MAX_NODES:
            pad_size = MAX_NODES - adj_target.size(1)
            adj_target = F.pad(adj_target, (0, pad_size, 0, pad_size))
            
        adj_target = adj_target.to(DEVICE)
        
        adj_pred, mu, logvar = model(data)
        
        loss, recon, kl = loss_function(adj_pred, adj_target, mu, logvar, BETA)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        total_recon += recon.item() * data.num_graphs
        total_kl += kl.item() * data.num_graphs
        
    return total_loss / len(loader.dataset), total_recon / len(loader.dataset), total_kl / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss, total_recon, total_kl = 0, 0, 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            adj_target = to_dense_adj(data.edge_index, data.batch, max_num_nodes=MAX_NODES)
            
            if adj_target.size(1) < MAX_NODES:
                pad_size = MAX_NODES - adj_target.size(1)
                adj_target = F.pad(adj_target, (0, pad_size, 0, pad_size))
                
            adj_target = adj_target.to(DEVICE)
            adj_pred, mu, logvar = model(data)
            loss, recon, kl = loss_function(adj_pred, adj_target, mu, logvar, BETA)
            
            total_loss += loss.item() * data.num_graphs
            total_recon += recon.item() * data.num_graphs
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
    
    model = GraphVAE(NODE_FEATURES, HIDDEN_DIM, LATENT_DIM, MAX_NODES).to(DEVICE)
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