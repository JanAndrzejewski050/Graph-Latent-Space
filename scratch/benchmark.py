import time
import torch
from torch_geometric.loader import DataLoader
import os

from src.config import TrainConfig
from src.model_registry import create_model
from src.train import _prepare_features, _prepare_adj_target_multibond, _prepare_node_targets

config = TrainConfig(arch="enhanced_gcn_vae", use_multi_bond=True, batch_size=256)
device = torch.device("mps")
model = create_model(config.arch, config).to(device)

train_dataset = torch.load(config.abs_path(config.train_data), weights_only=False)
train_dataset = [g for g in train_dataset if g.num_nodes <= config.max_nodes]

# Test DataLoader with num_workers=0 vs 4
for nw in [0, 4]:
    print(f"\n--- Testing num_workers={nw} ---")
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                        num_workers=nw, persistent_workers=nw > 0)
    
    t0 = time.time()
    for idx, data in enumerate(loader):
        if idx >= 10: break
        
        t1 = time.time()
        data = _prepare_features(data, config, device)
        batch_size = data.batch.max().item() + 1
        
        t2 = time.time()
        adj_target = _prepare_adj_target_multibond(data, batch_size, config.max_nodes, device)
        node_target, node_mask = _prepare_node_targets(data, batch_size, config.max_nodes, device)
        
        t3 = time.time()
        adj_pred, node_pred, mu, logvar = model(data)
        
        t4 = time.time()
        loss = adj_pred.sum() + node_pred.sum()
        loss.backward()
        t5 = time.time()
        
        print(f"Batch {idx}: Load={(t1-t0)*1000:.1f}ms, Prep={(t3-t2)*1000:.1f}ms, Forward={(t4-t3)*1000:.1f}ms, Back={(t5-t4)*1000:.1f}ms")
        t0 = time.time()
