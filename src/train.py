import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from rdkit import Chem

from src.config import (
    TrainConfig,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    ATOM_TO_IDX,
    BOND_TYPES,
    SUPPORTED_ATOMS,
    get_device,
    get_project_root,
)
from src.model_registry import create_model

import src.models  


def loss_function_binary(adj_pred, adj_target, node_pred, node_target,
                         node_mask, mu, logvar, beta):
    """Loss for models with binary adjacency prediction."""
    # Adjacency reconstruction (binary cross-entropy with logits)
    recon_adj = F.binary_cross_entropy_with_logits(
        adj_pred, adj_target, reduction="mean"
    )

    # Node type reconstruction (cross-entropy, masked)
    node_pred_flat = node_pred.view(-1, NUM_ATOM_TYPES)
    node_target_flat = node_target.view(-1)
    mask_flat = node_mask.view(-1).bool()

    if mask_flat.sum() > 0:
        recon_node = F.cross_entropy(
            node_pred_flat[mask_flat], node_target_flat[mask_flat],
            reduction="mean"
        )
    else:
        recon_node = torch.tensor(0.0, device=adj_pred.device)

    # KL divergence
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_adj + recon_node + beta * kl
    return total, recon_adj.item(), recon_node.item(), kl.item()


def loss_function_multibond(adj_pred, adj_target, node_pred, node_target,
                            node_mask, mu, logvar, beta):
    """
    Loss for models with multi-bond type prediction.

    adj_pred:   [B, num_bond_types, N, N]  (logits)
    adj_target: [B, N, N]  long tensor with bond-type indices
    """
    B, C, N, _ = adj_pred.shape

    # Reshape for cross-entropy: [B*N*N, C]
    adj_pred_flat = adj_pred.permute(0, 2, 3, 1).reshape(-1, C)
    adj_target_flat = adj_target.reshape(-1).long()

    # Class weights: bonds are rare compared to "no bond"
    recon_adj = F.cross_entropy(adj_pred_flat, adj_target_flat, reduction="mean")

    # Node type reconstruction (same as binary)
    node_pred_flat = node_pred.view(-1, NUM_ATOM_TYPES)
    node_target_flat = node_target.view(-1)
    mask_flat = node_mask.view(-1).bool()

    if mask_flat.sum() > 0:
        recon_node = F.cross_entropy(
            node_pred_flat[mask_flat], node_target_flat[mask_flat],
            reduction="mean"
        )
    else:
        recon_node = torch.tensor(0.0, device=adj_pred.device)

    # KL divergence
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_adj + recon_node + beta * kl
    return total, recon_adj.item(), recon_node.item(), kl.item()


def _prepare_node_targets(data, batch_size, max_nodes, device):
    """Build padded node-type targets and mask from a data batch."""
    node_target = torch.zeros(batch_size, max_nodes, dtype=torch.long, device=device)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)

    current_idx = 0
    for i in range(batch_size):
        num_nodes = (data.batch == i).sum().item()
        n = min(num_nodes, max_nodes)
        for j in range(n):
            # Determine atom type: if one-hot, take argmax; else take first col
            if data.x.size(1) >= NUM_ATOM_TYPES:
                # One-hot encoded: take argmax of first NUM_ATOM_TYPES columns
                atom_idx = data.x[current_idx + j, :NUM_ATOM_TYPES].argmax().item()
            else:
                # Raw atomic number
                atomic_num = int(data.x[current_idx + j][0].item())
                atom_idx = ATOM_TO_IDX.get(atomic_num, 1)  # default Carbon
            node_target[i, j] = atom_idx
            node_mask[i, j] = True
        current_idx += num_nodes

    return node_target, node_mask


def _prepare_adj_target_binary(data, batch_size, max_nodes, device):
    """Build padded binary adjacency target from a data batch."""
    adj = to_dense_adj(data.edge_index, data.batch, max_num_nodes=max_nodes)
    if adj.size(1) < max_nodes:
        pad = max_nodes - adj.size(1)
        adj = F.pad(adj, (0, pad, 0, pad))
    return adj.to(device)


def _prepare_adj_target_multibond(data, batch_size, max_nodes, device):
    adj = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.long, device=device)

    edge_index = data.edge_index 
    batch_vec = data.batch       

    
    num_nodes_per_graph = torch.zeros(batch_size, dtype=torch.long, device=device)
    for i in range(batch_size):
        num_nodes_per_graph[i] = (batch_vec == i).sum()

    cumulative = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
    cumulative[1:] = torch.cumsum(num_nodes_per_graph, dim=0)

    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    src_batch = batch_vec[src_nodes]

    # Local indices
    offsets = cumulative[src_batch]
    local_src = src_nodes - offsets
    local_dst = dst_nodes - offsets

    # Clamp to max_nodes
    valid = (local_src < max_nodes) & (local_dst < max_nodes)
    local_src = local_src[valid]
    local_dst = local_dst[valid]
    graph_ids = src_batch[valid]

    # Default bond type: SINGLE (index 1)
    bond_idx = BOND_TYPES["SINGLE"]

    # If edge_attr exists and has bond type info, use it
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        # Assume edge_attr[:, 0] encodes bond type somehow
        # For now, use SINGLE as default
        pass

    adj[graph_ids, local_src, local_dst] = bond_idx

    return adj


def _prepare_features(data, config, device):
    """
    Optionally slice node features to one-hot atom types only.
    """
    data = data.to(device)
    if config.use_onehot_only and data.x.size(1) > NUM_ATOM_TYPES:
        data.x = data.x[:, :NUM_ATOM_TYPES]
    return data



def train_epoch(model, loader, optimizer, config, beta, device):
    model.train()
    total_loss = 0.0
    total_adj = 0.0
    total_node = 0.0
    total_kl = 0.0
    total_graphs = 0

    is_multi = model.is_multi_bond
    loss_fn = loss_function_multibond if is_multi else loss_function_binary

    for data in loader:
        data = _prepare_features(data, config, device)
        batch_size = data.batch.max().item() + 1
        optimizer.zero_grad()

        # Targets
        if is_multi:
            adj_target = _prepare_adj_target_multibond(data, batch_size, config.max_nodes, device)
        else:
            adj_target = _prepare_adj_target_binary(data, batch_size, config.max_nodes, device)

        node_target, node_mask = _prepare_node_targets(data, batch_size, config.max_nodes, device)

        # Forward
        adj_pred, node_pred, mu, logvar = model(data)

        loss, adj_l, node_l, kl_l = loss_fn(
            adj_pred, adj_target, node_pred, node_target, node_mask,
            mu, logvar, beta
        )

        loss.backward()

        # Gradient clipping
        if config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

        optimizer.step()

        n = data.num_graphs
        total_loss += loss.item() * n
        total_adj += adj_l * n
        total_node += node_l * n
        total_kl += kl_l * n
        total_graphs += n

    return {
        "loss": total_loss / total_graphs,
        "adj_loss": total_adj / total_graphs,
        "node_loss": total_node / total_graphs,
        "kl_loss": total_kl / total_graphs,
    }


@torch.no_grad()
def eval_epoch(model, loader, config, beta, device):
    model.eval()
    total_loss = 0.0
    total_adj = 0.0
    total_node = 0.0
    total_kl = 0.0
    total_graphs = 0

    is_multi = model.is_multi_bond
    loss_fn = loss_function_multibond if is_multi else loss_function_binary

    for data in loader:
        data = _prepare_features(data, config, device)
        batch_size = data.batch.max().item() + 1

        if is_multi:
            adj_target = _prepare_adj_target_multibond(data, batch_size, config.max_nodes, device)
        else:
            adj_target = _prepare_adj_target_binary(data, batch_size, config.max_nodes, device)

        node_target, node_mask = _prepare_node_targets(data, batch_size, config.max_nodes, device)

        adj_pred, node_pred, mu, logvar = model(data)

        loss, adj_l, node_l, kl_l = loss_fn(
            adj_pred, adj_target, node_pred, node_target, node_mask,
            mu, logvar, beta
        )

        n = data.num_graphs
        total_loss += loss.item() * n
        total_adj += adj_l * n
        total_node += node_l * n
        total_kl += kl_l * n
        total_graphs += n

    return {
        "loss": total_loss / total_graphs,
        "adj_loss": total_adj / total_graphs,
        "node_loss": total_node / total_graphs,
        "kl_loss": total_kl / total_graphs,
    }



def run_training(config: TrainConfig):
    """Full training run with the given configuration."""
    device = get_device()
    print(f"Using device: {device}")

    train_dataset = torch.load(config.abs_path(config.train_data), weights_only=False)
    val_dataset = torch.load(config.abs_path(config.val_data), weights_only=False)

    train_dataset = [g for g in train_dataset if g.num_nodes <= config.max_nodes]
    val_dataset = [g for g in val_dataset if g.num_nodes <= config.max_nodes]

    print(f"Training on {len(train_dataset)} graphs, validating on {len(val_dataset)}")

    # Disable multiprocessing workers on macOS to avoid 'spawn' overhead/crashes
    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=num_workers)

    model = create_model(config.arch, config).to(device)
    print(f"Architecture: {config.arch}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    out_dir = config.abs_path(os.path.join(config.output_dir, config.arch))
    os.makedirs(out_dir, exist_ok=True)

    history = {
        "train_loss": [], "train_adj_loss": [], "train_node_loss": [], "train_kl_loss": [],
        "val_loss": [], "val_adj_loss": [], "val_node_loss": [], "val_kl_loss": [],
        "beta": [], "lr": [],
    }
    best_val_loss = float("inf")

    t0 = time.time()
    for epoch in range(1, config.epochs + 1):
        beta = config.get_beta(epoch)
        lr = optimizer.param_groups[0]["lr"]

        t_metrics = train_epoch(model, train_loader, optimizer, config, beta, device)
        v_metrics = eval_epoch(model, val_loader, config, beta, device)

        scheduler.step(v_metrics["loss"])

        for k in ["loss", "adj_loss", "node_loss", "kl_loss"]:
            history[f"train_{k}"].append(t_metrics[k])
            history[f"val_{k}"].append(v_metrics[k])
        history["beta"].append(beta)
        history["lr"].append(lr)

        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} | "
            f"β={beta:.4f} | lr={lr:.2e} | "
            f"Train Loss: {t_metrics['loss']:.4f} "
            f"(Adj: {t_metrics['adj_loss']:.4f}, "
            f"Node: {t_metrics['node_loss']:.4f}, "
            f"KL: {t_metrics['kl_loss']:.4f}) | "
            f"Val Loss: {v_metrics['loss']:.4f}"
        )

        if v_metrics["loss"] < best_val_loss:
            best_val_loss = v_metrics["loss"]
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))

        if config.save_every > 0 and epoch % config.save_every == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"checkpoint_epoch{epoch}.pth"))

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s. Best val loss: {best_val_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(out_dir, "final_model.pth"))

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    print(f"Outputs saved to {out_dir}")
    return model, history



def parse_args():
    parser = argparse.ArgumentParser(description="Train a Graph VAE model")
    parser.add_argument("--arch", type=str, default="enhanced_gcn_vae",
                        help="Architecture name (e.g., enhanced_gcn_vae, gcn_vae_v1)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--beta_end", type=float, default=None)
    parser.add_argument("--beta_warmup", type=int, default=None)
    parser.add_argument("--no_multi_bond", action="store_true",
                        help="Disable multi-bond prediction (use binary adj)")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(arch=args.arch)

    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.latent_dim is not None:
        config.latent_dim = args.latent_dim
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    if args.beta_end is not None:
        config.beta_end = args.beta_end
    if args.beta_warmup is not None:
        config.beta_warmup_epochs = args.beta_warmup
    if args.no_multi_bond:
        config.use_multi_bond = False
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    run_training(config)


if __name__ == "__main__":
    main()
