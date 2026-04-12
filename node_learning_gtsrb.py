"""
Node Learning on GTSRB (German Traffic Sign Recognition Benchmark)
==================================================================

Simulates decentralised learning where 4 camera nodes observe the same traffic
signs through different degradations:

    1. Cheap Sensor     — low-res blur + noise   (budget doorbell/IP cam)
    2. Grayscale / IR   — monochrome + noise      (IR camera, B&W CCTV)
    3. Colour Shifted   — bad white balance        (sodium/fluorescent/tungsten)
    4. Normal Camera    — no degradation           (decent vehicle dashcam)

Each node trains its own CNN encoder + classifier independently on its
distorted view of the GTSRB training set (43 traffic sign classes, ~39K images,
resized to 48x48).

A cross-attention fusion head (transformer-style) is then trained on frozen
node embeddings. During collaborative inference, when a node's confidence
(normalised entropy) falls below a threshold, it requests embeddings from
peers and the fusion head combines them via multi-head self-attention.

Regularisation:
    - L2-normalised embeddings before attention (prevents magnitude bias)
    - Entropy penalty on pooling weights (prevents attention collapse)

Metrics reported:
    - Solo accuracy per node and clean baseline
    - Fusion-always accuracy + average attention weights
    - Collaborative inference: triggered rate, fixed/broke/kept stats
    - Collaboration Efficiency (CE) from the Node Learning paper (Sec 3.5)

"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from distortions_final import (
    cheap_sensor, grayscale_ir, colour_shifted, normal_camera,
    stereo_shift, random_occlusion,
    horizontal_sign_angle, vertical_sign_angle,
    sign_approaching, overcast_flat_light,
    NodeTransform, RandomTransformChain, CAMERA_PRESETS,
)

# ============================================================================
# Config
# ============================================================================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
EMBEDDING_DIM = 128
NUM_CLASSES = 43
CONFIDENCE_THRESHOLD = 0.6
FUSION_EPOCHS = 20
ENTROPY_LAMBDA = 0.05        # weight for attention entropy regularization
MODEL_DIR = "./models"
DATA_DIR = "./GTSRB_data"
IMG_SIZE = 48

print(f"Using device: {DEVICE}")


# ============================================================================
# Dataset — loads GTSRB images, resizes to 48x48, applies distortion
# ============================================================================
class DistortedGTSRB(Dataset):
    """
    GTSRB dataset with optional distortion and geometric augmentation.

    Args:
        csv_path: Path to Train.csv or Test.csv
        data_dir: Directory containing the image files
        distortion_fn: (Legacy) Numpy-based distortion function
        transform: (New) NodeTransform instance for combined distortion + augmentation

    Note: If both distortion_fn and transform are provided, transform takes precedence.
    """
    def __init__(self, csv_path, data_dir, distortion_fn=None, transform=None):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.distortion_fn = distortion_fn
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.data_dir, row["Path"])
        label = int(row["ClassId"])

        # load and resize to fixed size
        img = Image.open(path).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR
        )
        img = np.array(img)  # (48, 48, 3) uint8

        if self.transform is not None:
            # New path: NodeTransform handles everything and returns tensor
            img = self.transform(img)
        else:
            # Legacy path: apply distortion_fn then manual conversion
            if self.distortion_fn is not None:
                img = self.distortion_fn(img)
            # HWC uint8 -> CHW float32, normalize to [0, 1]
            img = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(label, dtype=torch.long)
        return img, label


# ============================================================================
# MultiResolutionDataset — serves same image at different resolutions per node
# ============================================================================
class MultiResolutionDataset(Dataset):
    """
    Serves the same image at different resolutions for each node.
    Used for training the fusion head with multi-resolution nodes.

    Args:
        csv_path: Path to Train.csv or Test.csv
        data_dir: Directory containing the image files
        node_configs: Dict mapping node_name to {"size": int, "transform": NodeTransform}

    Returns:
        images: dict[str, Tensor]  # {node_name: tensor at node's resolution}
        label: Tensor

    Example:
        node_configs = {
            "cheap": {"size": 48, "transform": NodeTransform(cheap_sensor, "handheld", 48)},
            "ir": {"size": 48, "transform": NodeTransform(grayscale_ir, "pole_fixed", 48)},
        }
        dataset = MultiResolutionDataset(train_csv, DATA_DIR, node_configs)
    """
    def __init__(self, csv_path, data_dir, node_configs):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.node_configs = node_configs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.data_dir, row["Path"])
        label = int(row["ClassId"])

        # Load original image (don't resize yet)
        original = Image.open(path).convert("RGB")

        images = {}
        for node_name, config in self.node_configs.items():
            # Resize to this node's resolution
            size = config["size"]
            img = original.resize((size, size), Image.BILINEAR)
            img = np.array(img)

            # Apply node's transform
            transform = config.get("transform")
            if transform is not None:
                img = transform(img)
            else:
                img = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0

            images[node_name] = img

        label = torch.tensor(label, dtype=torch.long)
        return images, label


# ============================================================================
# Model — CNN encoder + classifier (adapted for 48x48 input)
# ============================================================================
class Encoder(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 24 -> 12
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 12 -> 6
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 6 -> 1x1
        )
        self.projection = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return x


class Classifier(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, embedding):
        return self.head(embedding)


from merge_operators import FusionHead


class NodeModel(nn.Module):
    """
    A node in the decentralized network.

    Each node has:
    - encoder: extracts embeddings from images
    - classifier: classifies embeddings (solo mode)
    - fusion_head: combines own + peer embeddings (collaborative mode)
    """
    def __init__(self, include_fusion=True):
        super().__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()
        self.fusion_head = FusionHead() if include_fusion else None

    def forward(self, x):
        """Solo inference - just encode and classify."""
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits, embedding

    def fused_forward(self, own_embedding, peer_embeddings, padding_mask=None):
        """
        Collaborative inference - fuse own embedding with peers and classify.

        Args:
            own_embedding: (batch, embed_dim) - this node's embedding
            peer_embeddings: list of (batch, embed_dim) - peer embeddings
            padding_mask: (batch, N_nodes) - optional mask for dropped peers

        Returns:
            logits: (batch, num_classes)
            attn_weights: (batch, N_nodes)
        """
        if self.fusion_head is None:
            raise ValueError("This node has no fusion head. Initialize with include_fusion=True")

        # Stack: own embedding first, then peers
        all_embs = [own_embedding] + list(peer_embeddings)
        stacked = torch.stack(all_embs, dim=1)  # (batch, N_nodes, embed_dim)

        fused, attn_weights = self.fusion_head(stacked, padding_mask)
        logits = self.classifier(fused)

        return logits, attn_weights


# ============================================================================
# Cross-Attention Fusion (Transformer-style) - LEGACY, kept for compatibility
# ============================================================================
class CrossAttentionFusion(nn.Module):
    """
    Transformer-style cross-attention fusion for node embeddings.

    Each node's embedding attends to all other nodes via Q/K/V, allowing
    pairwise interaction before fusion. This captures relationships like
    "night_cam has shape info that motion_blur lacks" — not just individual
    importance scores.

    Architecture:
        1. L2-normalize embeddings (remove magnitude bias)
        2. One transformer encoder layer (multi-head self-attention + FFN)
        3. Mean-pool the refined tokens → single fused embedding
        4. Classify

    Returns logits and per-node attention weights for interpretability.
    """
    def __init__(self, embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES,
                 n_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        # transformer encoder layer: self-attention + feed-forward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # post-attention scoring for interpretability + entropy regularization
        self.attn_score = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # classifier on the fused embedding
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, embeddings):
        # embeddings: (batch, N_nodes, embed_dim)
        # L2-normalize to remove magnitude bias between encoders
        normed = nn.functional.normalize(embeddings, p=2, dim=-1)

        # cross-attention: each node attends to all others
        refined = self.transformer(normed)              # (batch, N, embed_dim)

        # compute interpretable attention weights for pooling
        scores = self.attn_score(refined)               # (batch, N, 1)
        weights = torch.softmax(scores, dim=1)          # (batch, N, 1)

        # weighted pool → fused embedding
        fused = (weights * refined).sum(dim=1)          # (batch, embed_dim)
        logits = self.classifier(fused)                 # (batch, num_classes)
        return logits, weights.squeeze(-1)              # logits, (batch, N)


# ============================================================================
# Confidence measure — normalized entropy
# ============================================================================
def confidence_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=-1)
    max_entropy = np.log(probs.size(-1))
    return 1.0 - entropy / max_entropy


confidence_fn = confidence_entropy


# ============================================================================
# Training
# ============================================================================
def train_node(node_model, train_loader, node_name):
    node_model.to(DEVICE)
    optimizer = optim.Adam(node_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    epoch_bar = tqdm(range(EPOCHS), desc=f"  {node_name}", unit="ep")
    for epoch in epoch_bar:
        node_model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits, _ = node_model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        acc = correct / total * 100
        avg_loss = total_loss / total
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.1f}%")

    print(f"  [{node_name}] Final — Loss: {avg_loss:.4f}  Acc: {acc:.1f}%")
    return node_model


# ============================================================================
# Training — cross-attention fusion head
# ============================================================================
def train_fusion(fusion_model, nodes, multi_res_loader, node_names):
    """
    Train the cross-attention fusion head on stacked embeddings from all nodes.
    Node encoders are frozen — only the fusion module learns.

    Args:
        fusion_model: CrossAttentionFusion model
        nodes: Dict of trained NodeModel instances
        multi_res_loader: DataLoader returning (images_dict, labels) from MultiResolutionDataset
        node_names: List of node names (determines stacking order)
    """
    fusion_model.to(DEVICE)
    for node in nodes.values():
        node.eval()

    optimizer = optim.Adam(fusion_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    epoch_bar = tqdm(range(FUSION_EPOCHS), desc="  fusion", unit="ep")
    for epoch in epoch_bar:
        fusion_model.train()
        total_loss, correct, total = 0, 0, 0

        for images_dict, labels in multi_res_loader:
            labels = labels.to(DEVICE)

            # Extract embeddings from each node's encoder at its native resolution
            emb_list = []
            for name in node_names:
                images = images_dict[name].to(DEVICE)
                with torch.no_grad():
                    emb = nodes[name].encoder(images)
                emb_list.append(emb)

            stacked = torch.stack(emb_list, dim=1)  # (batch, N_nodes, embed_dim)

            logits, attn_weights = fusion_model(stacked)
            ce_loss = criterion(logits, labels)

            # entropy regularization: penalize peaked attention
            attn_entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean()
            loss = ce_loss - ENTROPY_LAMBDA * attn_entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        acc = correct / total * 100
        avg_loss = total_loss / total
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.1f}%")

    print(f"  [fusion] Final — Loss: {avg_loss:.4f}  Acc: {acc:.1f}%")
    return fusion_model


def train_fusion_with_dropout(fusion_model, nodes, multi_res_loader, node_names, p_drop=0.0):
    """
    Train the cross-attention fusion head with optional node dropout.

    During training, randomly zeros out node embeddings with probability p_drop.
    This teaches the fusion head to handle missing peers gracefully.

    Args:
        fusion_model: CrossAttentionFusion model
        nodes: Dict of trained NodeModel instances
        multi_res_loader: DataLoader returning (images_dict, labels)
        node_names: List of node names
        p_drop: Probability of dropping each node per sample (0.0 = no dropout)
    """
    fusion_model.to(DEVICE)
    for node in nodes.values():
        node.eval()

    optimizer = optim.Adam(fusion_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    n_nodes = len(node_names)

    epoch_bar = tqdm(range(FUSION_EPOCHS), desc="  fusion+dropout", unit="ep")
    for epoch in epoch_bar:
        fusion_model.train()
        total_loss, correct, total = 0, 0, 0

        for images_dict, labels in multi_res_loader:
            batch_size = labels.size(0)
            labels = labels.to(DEVICE)

            # Extract embeddings from each node's encoder
            emb_list = []
            for name in node_names:
                images = images_dict[name].to(DEVICE)
                with torch.no_grad():
                    emb = nodes[name].encoder(images)
                emb_list.append(emb)

            stacked = torch.stack(emb_list, dim=1)  # (batch, N, embed_dim)

            # Apply node dropout during training
            if p_drop > 0 and fusion_model.training:
                # Random mask: 1 = keep, 0 = drop
                mask = (torch.rand(batch_size, n_nodes, device=DEVICE) > p_drop).float()

                # Ensure at least 2 nodes are kept per sample
                keep_counts = mask.sum(dim=1)
                for i in range(batch_size):
                    if keep_counts[i] < 2:
                        keep_indices = torch.randperm(n_nodes, device=DEVICE)[:2]
                        mask[i] = 0
                        mask[i, keep_indices] = 1

                # Apply mask: zero out dropped embeddings
                mask = mask.unsqueeze(-1)  # (batch, N, 1)
                stacked = stacked * mask

                # Create padding mask for transformer (True = ignore)
                padding_mask = (mask.squeeze(-1) == 0)
            else:
                padding_mask = None

            # Forward through fusion with optional mask
            normed = nn.functional.normalize(stacked, p=2, dim=-1)
            refined = fusion_model.transformer(normed, src_key_padding_mask=padding_mask)
            scores = fusion_model.attn_score(refined)

            # Mask attention scores before softmax
            if padding_mask is not None:
                scores = scores.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))

            weights = torch.softmax(scores, dim=1)
            fused = (weights * refined).sum(dim=1)
            logits = fusion_model.classifier(fused)

            ce_loss = criterion(logits, labels)
            attn_entropy = -(weights.squeeze(-1) * torch.log(weights.squeeze(-1) + 1e-8)).sum(dim=-1).mean()
            loss = ce_loss - ENTROPY_LAMBDA * attn_entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            correct += (logits.argmax(1) == labels).sum().item()
            total += batch_size

        acc = correct / total * 100
        avg_loss = total_loss / total
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.1f}%", p_drop=f"{p_drop:.1f}")

    print(f"  [fusion+dropout] Final — Loss: {avg_loss:.4f}  Acc: {acc:.1f}%  (p_drop={p_drop})")
    return fusion_model


def train_node_fusion_heads(nodes, multi_res_loader, node_names, p_drop=0.0):
    """
    Train each node's fusion head from its own perspective.

    Each node learns to combine its own embedding with peer embeddings.
    The node's own embedding is always first in the stack.

    Args:
        nodes: Dict of NodeModel instances (must have fusion_head)
        multi_res_loader: DataLoader returning (images_dict, labels)
        node_names: List of node names
        p_drop: Probability of dropping each peer per sample
    """
    # Freeze encoders, only train fusion heads
    for node in nodes.values():
        node.encoder.eval()
        for param in node.encoder.parameters():
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    n_nodes = len(node_names)

    for requesting_name in node_names:
        print(f"\n  Training fusion head for: {requesting_name}")
        requesting_node = nodes[requesting_name]
        requesting_node.fusion_head.train()

        # Only optimize this node's fusion head (classifier stays frozen)
        optimizer = optim.Adam(requesting_node.fusion_head.parameters(), lr=LR)

        peer_names = [n for n in node_names if n != requesting_name]

        epoch_bar = tqdm(range(FUSION_EPOCHS), desc=f"    {requesting_name}", unit="ep")
        for epoch in epoch_bar:
            total_loss, correct, total = 0, 0, 0

            for images_dict, labels in multi_res_loader:
                batch_size = labels.size(0)
                labels = labels.to(DEVICE)

                # Get requesting node's embedding
                with torch.no_grad():
                    own_images = images_dict[requesting_name].to(DEVICE)
                    own_emb = requesting_node.encoder(own_images)

                # Get peer embeddings
                peer_embs = []
                for peer_name in peer_names:
                    with torch.no_grad():
                        peer_images = images_dict[peer_name].to(DEVICE)
                        peer_emb = nodes[peer_name].encoder(peer_images)
                    peer_embs.append(peer_emb)

                # Apply peer dropout (own embedding is never dropped)
                if p_drop > 0:
                    # Mask: True = dropped
                    peer_mask = torch.rand(batch_size, len(peer_names), device=DEVICE) < p_drop
                    # Ensure at least 1 peer is available
                    all_dropped = peer_mask.all(dim=1)
                    for i in range(batch_size):
                        if all_dropped[i]:
                            keep_idx = torch.randint(len(peer_names), (1,)).item()
                            peer_mask[i, keep_idx] = False

                    # Full mask includes own embedding (never dropped)
                    padding_mask = torch.cat([
                        torch.zeros(batch_size, 1, device=DEVICE, dtype=torch.bool),
                        peer_mask
                    ], dim=1)
                else:
                    padding_mask = None

                # Forward through this node's fusion head
                logits, attn_weights = requesting_node.fused_forward(
                    own_emb, peer_embs, padding_mask
                )

                # Loss with entropy regularization
                ce_loss = criterion(logits, labels)
                attn_entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean()
                loss = ce_loss - ENTROPY_LAMBDA * attn_entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_size
                correct += (logits.argmax(1) == labels).sum().item()
                total += batch_size

            acc = correct / total * 100
            avg_loss = total_loss / total
            epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.1f}%")

        print(f"    [{requesting_name}] Fusion Final — Loss: {avg_loss:.4f}  Acc: {acc:.1f}%")

    # Unfreeze encoders for future use
    for node in nodes.values():
        for param in node.encoder.parameters():
            param.requires_grad = True

    return nodes


# ============================================================================
# Collaborative Inference — cross-attention fusion
# ============================================================================
@torch.no_grad()
def collaborative_inference(nodes, multi_res_loader):
    """
    Collaborative inference using each node's own fusion head.

    Args:
        nodes: Dict of NodeModel instances (with fusion_head)
        multi_res_loader: DataLoader returning (images_dict, labels)
    """
    for node in nodes.values():
        node.eval()

    node_names = list(nodes.keys())

    all_embeddings = {name: [] for name in nodes}
    all_logits = {name: [] for name in nodes}
    all_labels = []

    for images_dict, labels in multi_res_loader:
        for name in node_names:
            images = images_dict[name].to(DEVICE)
            logits, embeddings = nodes[name](images)
            all_embeddings[name].append(embeddings.cpu())
            all_logits[name].append(logits.cpu())
        all_labels.append(labels)

    for name in nodes:
        all_embeddings[name] = torch.cat(all_embeddings[name])
        all_logits[name] = torch.cat(all_logits[name])
    all_labels = torch.cat(all_labels)

    n_samples = len(all_labels)

    stats = {name: {
        "solo_correct": 0,
        "collab_correct": 0,
        "collab_triggered": 0,
        "collab_solo_was_wrong": 0,
        "collab_solo_was_right": 0,
        "collab_fixed": 0,
        "collab_broke": 0,
        "collab_stayed_wrong": 0,
        "collab_stayed_right": 0,
        "no_collab_correct": 0,
        "no_collab_wrong": 0,
        "confidences_when_triggered": [],
        "confidences_when_not_triggered": [],
        "attn_weights_when_triggered": [],
    } for name in nodes}

    for i in range(n_samples):
        label = all_labels[i].item()

        confidences = {}
        for name in node_names:
            conf = confidence_fn(all_logits[name][i].unsqueeze(0)).item()
            confidences[name] = conf

        for name in node_names:
            s = stats[name]
            solo_pred = all_logits[name][i].argmax().item()
            solo_is_correct = (solo_pred == label)

            if solo_is_correct:
                s["solo_correct"] += 1

            if confidences[name] >= CONFIDENCE_THRESHOLD:
                pred = solo_pred
                s["confidences_when_not_triggered"].append(confidences[name])
                if pred == label:
                    s["no_collab_correct"] += 1
                else:
                    s["no_collab_wrong"] += 1
            else:
                s["collab_triggered"] += 1
                s["confidences_when_triggered"].append(confidences[name])

                # Use THIS node's fusion head to combine with peers
                own_emb = all_embeddings[name][i].unsqueeze(0).to(DEVICE)
                peer_embs = [all_embeddings[p][i].unsqueeze(0).to(DEVICE)
                            for p in node_names if p != name]

                fused_logits, attn_weights = nodes[name].fused_forward(own_emb, peer_embs)
                fused_pred = fused_logits.argmax(1).item()
                attn_w = attn_weights.squeeze(0).cpu().tolist()
                s["attn_weights_when_triggered"].append(attn_w)

                pred = fused_pred
                collab_is_correct = (pred == label)

                if solo_is_correct:
                    s["collab_solo_was_right"] += 1
                    if collab_is_correct:
                        s["collab_stayed_right"] += 1
                    else:
                        s["collab_broke"] += 1
                else:
                    s["collab_solo_was_wrong"] += 1
                    if collab_is_correct:
                        s["collab_fixed"] += 1
                    else:
                        s["collab_stayed_wrong"] += 1

            if pred == label:
                s["collab_correct"] += 1

    return stats, n_samples


@torch.no_grad()
def collaborative_inference_with_dropout(
    nodes,
    multi_res_loader,
    merge_operator=None,
    p_drop=0.0,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    fusion_model=None,  # Legacy parameter, ignored if nodes have fusion_head
):
    """
    Collaborative inference using each node's own fusion head.

    Each node uses its own fusion head to combine its embedding with peers.
    This is the decentralized approach where nodes don't share a fusion model.

    Args:
        nodes: Dict of NodeModel instances (with fusion_head)
        multi_res_loader: DataLoader returning (images_dict, labels)
        merge_operator: Optional MergeOperator (overrides node's fusion_head)
        p_drop: Probability of each peer being unavailable (0.0 = all available)
        confidence_threshold: Threshold for triggering collaboration
        fusion_model: Legacy parameter for backwards compatibility

    Returns:
        stats: Dict of per-node statistics
        n_samples: Total number of samples evaluated
    """
    for node in nodes.values():
        node.eval()

    node_names = list(nodes.keys())

    # Collect all embeddings and logits
    all_embeddings = {name: [] for name in nodes}
    all_logits = {name: [] for name in nodes}
    all_labels = []

    for images_dict, labels in multi_res_loader:
        for name in node_names:
            images = images_dict[name].to(DEVICE)
            logits, embeddings = nodes[name](images)
            all_embeddings[name].append(embeddings.cpu())
            all_logits[name].append(logits.cpu())
        all_labels.append(labels)

    for name in nodes:
        all_embeddings[name] = torch.cat(all_embeddings[name])
        all_logits[name] = torch.cat(all_logits[name])
    all_labels = torch.cat(all_labels)

    n_samples = len(all_labels)

    stats = {name: {
        "solo_correct": 0,
        "collab_correct": 0,
        "collab_triggered": 0,
        "collab_solo_was_wrong": 0,
        "collab_solo_was_right": 0,
        "collab_fixed": 0,
        "collab_broke": 0,
        "collab_stayed_wrong": 0,
        "collab_stayed_right": 0,
        "no_collab_correct": 0,
        "no_collab_wrong": 0,
        "peers_dropped": 0,
        "no_peers_available": 0,
    } for name in nodes}

    for i in range(n_samples):
        label = all_labels[i].item()

        # Compute confidence for each node
        confidences = {}
        for name in node_names:
            conf = confidence_fn(all_logits[name][i].unsqueeze(0)).item()
            confidences[name] = conf

        for name in node_names:
            s = stats[name]
            solo_pred = all_logits[name][i].argmax().item()
            solo_is_correct = (solo_pred == label)

            if solo_is_correct:
                s["solo_correct"] += 1

            if confidences[name] >= confidence_threshold:
                pred = solo_pred
                if pred == label:
                    s["no_collab_correct"] += 1
                else:
                    s["no_collab_wrong"] += 1
            else:
                s["collab_triggered"] += 1

                # Simulate node dropout for peers
                available_peers = []
                peer_confs = []
                for peer_name in node_names:
                    if peer_name == name:
                        continue
                    # Each peer has p_drop chance of being unavailable
                    if p_drop > 0 and np.random.random() < p_drop:
                        s["peers_dropped"] += 1
                        continue
                    available_peers.append(peer_name)
                    peer_confs.append(confidences[peer_name])

                if len(available_peers) == 0:
                    # No peers available, fall back to solo
                    s["no_peers_available"] += 1
                    pred = solo_pred
                else:
                    # Use requesting node's own fusion head
                    requesting_node = nodes[name]
                    requesting_emb = all_embeddings[name][i].unsqueeze(0).to(DEVICE)
                    peer_embs = [all_embeddings[p][i].unsqueeze(0).to(DEVICE) for p in available_peers]

                    if merge_operator is not None:
                        # Use external merge operator
                        peer_contexts = {
                            "confidences": peer_confs,
                            "peer_names": available_peers,
                            "requesting_name": name,
                            "requesting_confidence": confidences[name],
                        }
                        merged = merge_operator.merge(requesting_emb, peer_embs, peer_contexts)
                        # Use requesting node's classifier
                        merged_logits = requesting_node.classifier(merged)
                    else:
                        # Use requesting node's own fusion head
                        merged_logits, _ = requesting_node.fused_forward(
                            requesting_emb, peer_embs
                        )

                    pred = merged_logits.argmax(1).item()

                collab_is_correct = (pred == label)

                if solo_is_correct:
                    s["collab_solo_was_right"] += 1
                    if collab_is_correct:
                        s["collab_stayed_right"] += 1
                    else:
                        s["collab_broke"] += 1
                else:
                    s["collab_solo_was_wrong"] += 1
                    if collab_is_correct:
                        s["collab_fixed"] += 1
                    else:
                        s["collab_stayed_wrong"] += 1

            if pred == label:
                s["collab_correct"] += 1

    return stats, n_samples


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    t_start = time.time()
    train_csv = os.path.join(DATA_DIR, "Train.csv")
    test_csv = os.path.join(DATA_DIR, "Test.csv")

    # --- Define nodes with full transforms (distortion + geometric augmentation) ---
    # Each node operates at a different resolution to simulate heterogeneous hardware
    NODE_CONFIGS = {
        "cheap": {"distortion_fn": cheap_sensor, "preset": "handheld", "size": 48},
        "ir": {"distortion_fn": grayscale_ir, "preset": "pole_fixed", "size": 48},
        "colour_shifted": {"distortion_fn": colour_shifted, "preset": "vehicle_mounted", "size": 48},
        "normal": {"distortion_fn": normal_camera, "preset": "vehicle_mounted", "size": 48},
    }

    # --- Random environmental transforms applied on top of camera distortions ---
    # Order: stereo -> angle -> zoom -> lighting -> occlusion (physical realism)
    # Format: (function, probability, kwargs)
    RANDOM_TRANSFORMS = {
        "cheap": [
            (stereo_shift,          0.2,  {"border_mode": "reflect"}),
            (horizontal_sign_angle, 0.3,  {}),
            (vertical_sign_angle,   0.3,  {}),
            (sign_approaching,      0.1,  {}),
            (overcast_flat_light,   0.3,  {}),
            (random_occlusion,      0.15, {}),
        ],
        "ir": [
            (stereo_shift,          0.3,  {"border_mode": "reflect"}),
            (horizontal_sign_angle, 0.2,  {}),
            (vertical_sign_angle,   0.2,  {}),
            (sign_approaching,      0.1,  {}),
            (overcast_flat_light,   0.1,  {}),
            (random_occlusion,      0.2,  {}),
        ],
        "colour_shifted": [
            (stereo_shift,          0.15, {"border_mode": "reflect"}),
            (horizontal_sign_angle, 0.3,  {}),
            (vertical_sign_angle,   0.3,  {}),
            (sign_approaching,      0.1,  {}),
            (overcast_flat_light,   0.4,  {}),
            (random_occlusion,      0.15, {}),
        ],
        "normal": [
            (stereo_shift,          0.15, {"border_mode": "reflect"}),
            (horizontal_sign_angle, 0.25, {}),
            (vertical_sign_angle,   0.25, {}),
            (sign_approaching,      0.1,  {}),
            (overcast_flat_light,   0.25, {}),
            (random_occlusion,      0.1,  {}),
        ],
    }
    node_names = list(NODE_CONFIGS.keys())

    # --- Create datasets and loaders with NodeTransform ---
    train_loaders = {}
    test_loaders = {}
    for name, config in NODE_CONFIGS.items():
        random_chain = RandomTransformChain(RANDOM_TRANSFORMS.get(name, []))
        train_transform = NodeTransform(
            distortion_fn=config["distortion_fn"],
            preset=config["preset"],
            output_size=config["size"],
            training=True,
            random_transforms=random_chain,
        )
        test_transform = NodeTransform(
            distortion_fn=config["distortion_fn"],
            preset=config["preset"],
            output_size=config["size"],
            training=False,
            random_transforms=random_chain,
        )
        train_ds = DistortedGTSRB(train_csv, DATA_DIR, transform=train_transform)
        test_ds = DistortedGTSRB(test_csv, DATA_DIR, transform=test_transform)
        train_loaders[name] = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=2)
        test_loaders[name] = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)} images, Test: {len(test_ds)} images")
    node_sizes = ", ".join(f"{n}={c['size']}px" for n, c in NODE_CONFIGS.items())
    print(f"Node resolutions: {node_sizes}, Classes: {NUM_CLASSES}")

    # --- Create MultiResolutionDataset for fusion training/inference ---
    def build_multi_res_configs(node_configs, training):
        """Build node_configs dict for MultiResolutionDataset."""
        result = {}
        for name, cfg in node_configs.items():
            random_chain = RandomTransformChain(RANDOM_TRANSFORMS.get(name, []))
            transform = NodeTransform(
                distortion_fn=cfg["distortion_fn"],
                preset=cfg["preset"],
                output_size=cfg["size"],
                training=training,
                random_transforms=random_chain,
            )
            result[name] = {"size": cfg["size"], "transform": transform}
        return result

    multi_res_train_configs = build_multi_res_configs(NODE_CONFIGS, training=True)
    multi_res_test_configs = build_multi_res_configs(NODE_CONFIGS, training=False)

    multi_res_train_ds = MultiResolutionDataset(train_csv, DATA_DIR, multi_res_train_configs)
    multi_res_test_ds = MultiResolutionDataset(test_csv, DATA_DIR, multi_res_test_configs)

    multi_res_train_loader = DataLoader(
        multi_res_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    multi_res_test_loader = DataLoader(
        multi_res_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # --- Train each node independently (or load from disk) ---
    # Each node has its own encoder, classifier, AND fusion head
    os.makedirs(MODEL_DIR, exist_ok=True)

    nodes = {}
    any_needs_fusion_training = False

    for name in NODE_CONFIGS:
        model_path = os.path.join(MODEL_DIR, f"gtsrb_{name}_with_fusion.pt")
        old_model_path = os.path.join(MODEL_DIR, f"gtsrb_{name}.pt")

        model = NodeModel(include_fusion=True)

        if os.path.exists(model_path):
            print(f"\nLoading saved model (with fusion): {name}")
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
        elif os.path.exists(old_model_path):
            # Load old model (encoder + classifier only), fusion head needs training
            print(f"\nLoading saved model (no fusion): {name}")
            old_state = torch.load(old_model_path, map_location=DEVICE)
            # Filter out fusion_head keys that don't exist in old model
            model_state = model.state_dict()
            for key in old_state:
                if key in model_state:
                    model_state[key] = old_state[key]
            model.load_state_dict(model_state)
            model.to(DEVICE)
            any_needs_fusion_training = True
        else:
            print(f"\nTraining node: {name}")
            model = train_node(model, train_loaders[name], name)
            torch.save(model.state_dict(), old_model_path)
            print(f"  Saved solo model: {old_model_path}")
            any_needs_fusion_training = True
        nodes[name] = model

    # --- Clean baseline (no distortion, minimal augmentation) ---
    clean_train_transform = NodeTransform(
        distortion_fn=None,
        preset="pole_fixed",
        output_size=IMG_SIZE,
        training=True,
    )
    clean_test_transform = NodeTransform(
        distortion_fn=None,
        preset="pole_fixed",
        output_size=IMG_SIZE,
        training=False,
    )
    clean_train_ds = DistortedGTSRB(train_csv, DATA_DIR, transform=clean_train_transform)
    clean_test_ds = DistortedGTSRB(test_csv, DATA_DIR, transform=clean_test_transform)
    clean_train_loader = DataLoader(clean_train_ds, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=2)
    clean_test_loader = DataLoader(clean_test_ds, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=2)

    clean_model_path = os.path.join(MODEL_DIR, "gtsrb_clean_baseline.pt")
    clean_model = NodeModel(include_fusion=True)
    if os.path.exists(clean_model_path):
        print("\nLoading saved model: clean_baseline")
        saved_state = torch.load(clean_model_path, map_location=DEVICE)
        # Handle old checkpoints that lack fusion_head keys
        model_state = clean_model.state_dict()
        for key in saved_state:
            if key in model_state:
                model_state[key] = saved_state[key]
        clean_model.load_state_dict(model_state)
        clean_model.to(DEVICE)
    else:
        print("\nTraining baseline (clean images):")
        clean_model = train_node(clean_model, clean_train_loader, "clean_baseline")
        torch.save(clean_model.state_dict(), clean_model_path)
        print(f"  Saved to {clean_model_path}")

    # --- Evaluate solo performance ---
    print("\n" + "=" * 60)
    print("SOLO PERFORMANCE (each node classifies alone)")
    print("=" * 60)

    for name, model in {**nodes, "clean_baseline": clean_model}.items():
        print(f"\nEvaluating solo performance: {name}")
        model.eval()
        loader = test_loaders.get(name, clean_test_loader)
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                print(f"  Batch {total//BATCH_SIZE + 1}/{len(loader)}", end="\r")
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits, _ = model(images)
                correct += (logits.argmax(1) == labels).sum().item()
                total += images.size(0)
        print(f"  {name:15s}  Accuracy: {correct/total*100:.1f}%")

    # --- Train each node's fusion head (if needed) ---
    if any_needs_fusion_training:
        print("\nTraining per-node fusion heads:")
        nodes = train_node_fusion_heads(nodes, multi_res_train_loader, node_names, p_drop=0.1)

        # Save updated models with fusion heads
        for name in node_names:
            model_path = os.path.join(MODEL_DIR, f"gtsrb_{name}_with_fusion.pt")
            torch.save(nodes[name].state_dict(), model_path)
            print(f"  Saved {name} to {model_path}")
    else:
        print("\nAll fusion heads already trained.")

    # --- Evaluate per-node fusion (always fuse) ---
    print("\n" + "=" * 60)
    print("PER-NODE FUSION PERFORMANCE (each node uses its own fusion head)")
    print("=" * 60)

    for node in nodes.values():
        node.eval()

    fusion_stats = {name: {"correct": 0, "total": 0, "attn_weights": []} for name in node_names}

    with torch.no_grad():
        for images_dict, labels in multi_res_test_loader:
            labels = labels.to(DEVICE)
            batch_size = labels.size(0)

            # Get all embeddings
            embeddings = {}
            for name in node_names:
                images = images_dict[name].to(DEVICE)
                embeddings[name] = nodes[name].encoder(images)

            # Each node fuses from its perspective
            for name in node_names:
                own_emb = embeddings[name]
                peer_embs = [embeddings[p] for p in node_names if p != name]

                logits, weights = nodes[name].fused_forward(own_emb, peer_embs)
                fusion_stats[name]["correct"] += (logits.argmax(1) == labels).sum().item()
                fusion_stats[name]["total"] += batch_size
                fusion_stats[name]["attn_weights"].append(weights.cpu())

    print("\n  Per-node fusion accuracy (using own fusion head):")
    for name in node_names:
        acc = fusion_stats[name]["correct"] / fusion_stats[name]["total"] * 100
        avg_weights = torch.cat(fusion_stats[name]["attn_weights"]).mean(dim=0)
        print(f"    {name:15s}  Accuracy: {acc:.1f}%  Attention: [self={avg_weights[0]:.3f}, peers={avg_weights[1:].tolist()}]")

    # --- Collaborative inference with threshold ---
    print("\n" + "=" * 60)
    print(f"COLLABORATIVE INFERENCE (threshold={CONFIDENCE_THRESHOLD})")
    print(f"Confidence measure: normalized entropy")
    print(f"Fusion method: per-node fusion heads (decentralized)")
    print("=" * 60)

    stats, n_samples = collaborative_inference(nodes, multi_res_test_loader)

    for name in nodes:
        s = stats[name]
        solo_acc = s["solo_correct"] / n_samples * 100
        collab_acc = s["collab_correct"] / n_samples * 100
        triggered = s["collab_triggered"]
        delta = collab_acc - solo_acc

        print(f"\n  --- {name} ---")
        print(f"  Solo accuracy:    {solo_acc:.1f}%")
        print(f"  Collab accuracy:  {collab_acc:.1f}%  (Δ {delta:+.1f}%)")
        print(f"  Communication triggered: {triggered}/{n_samples} "
              f"({triggered/n_samples*100:.1f}%)")

        if triggered > 0:
            print(f"\n  When collaboration WAS triggered ({triggered} samples):")
            print(f"    Solo was wrong → collab FIXED it:     {s['collab_fixed']:4d}  "
                  f"({s['collab_fixed']/triggered*100:5.1f}%)")
            print(f"    Solo was wrong → collab STAYED wrong: {s['collab_stayed_wrong']:4d}  "
                  f"({s['collab_stayed_wrong']/triggered*100:5.1f}%)")
            print(f"    Solo was right → collab KEPT it:      {s['collab_stayed_right']:4d}  "
                  f"({s['collab_stayed_right']/triggered*100:5.1f}%)")
            print(f"    Solo was right → collab BROKE it:     {s['collab_broke']:4d}  "
                  f"({s['collab_broke']/triggered*100:5.1f}%)")

            net = s["collab_fixed"] - s["collab_broke"]
            print(f"    Net effect: {net:+d} samples ({net/triggered*100:+.1f}% of triggered)")

            avg_conf_triggered = np.mean(s["confidences_when_triggered"])
            print(f"    Avg confidence when triggered: {avg_conf_triggered:.3f}")

            # average attention weights when this node triggered collab
            attn_arr = np.array(s["attn_weights_when_triggered"])
            avg_attn = attn_arr.mean(axis=0)
            print(f"    Avg attention weights when triggered:")
            attn_labels = [name] + [p for p in node_names if p != name]
            for j, lbl in enumerate(attn_labels):
                tag = " (self)" if j == 0 else ""
                print(f"      {lbl:15s}  {avg_attn[j]:.3f}{tag}")

        no_collab_total = s["no_collab_correct"] + s["no_collab_wrong"]
        if no_collab_total > 0:
            print(f"\n  When NO collaboration ({no_collab_total} samples):")
            print(f"    Correct: {s['no_collab_correct']:4d}  "
                  f"({s['no_collab_correct']/no_collab_total*100:.1f}%)")
            print(f"    Wrong:   {s['no_collab_wrong']:4d}  "
                  f"({s['no_collab_wrong']/no_collab_total*100:.1f}%)")
            avg_conf_not = np.mean(s["confidences_when_not_triggered"])
            print(f"    Avg confidence: {avg_conf_not:.3f}")

    # --- Collaboration Efficiency ---
    embedding_bytes = EMBEDDING_DIM * 4
    print("\n" + "=" * 60)
    print("COLLABORATION EFFICIENCY (CE = accuracy gain / bytes exchanged)")
    print("=" * 60)
    print(f"  Embedding size per request: {embedding_bytes} bytes")
    for name in nodes:
        s = stats[name]
        delta = s["collab_correct"] - s["solo_correct"]
        bytes_exchanged = s["collab_triggered"] * embedding_bytes * (len(nodes) - 1)
        if bytes_exchanged > 0:
            ce = delta / bytes_exchanged
            print(f"  {name:15s}  CE: {ce:.6f} correct_samples/byte  "
                  f"(total bytes exchanged: {bytes_exchanged:,})")

    elapsed = time.time() - t_start
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\nTotal elapsed time: {minutes}m {seconds}s")
