"""
Node Learning on GTSRB (German Traffic Sign Recognition Benchmark)
==================================================================

Simulates decentralised learning where 4 camera nodes observe the same traffic
signs through different degradations:

    1. Grayscale / IR   — monochrome + noise      (IR camera, B&W CCTV)
    2. Colour Shifted   — bad white balance        (sodium/fluorescent/tungsten)

Each node trains its own CNN encoder + classifier independently on its
distorted view of the GTSRB training set (43 traffic sign classes, ~39K images,
resized to 48x48).

Collaborative inference by cross-attention & modified-TATE framework. 

Metrics reported:
    - Solo accuracy per node and clean baseline
    - Collaborative inference: triggered rate, fixed/broke/kept stats
    - Collaboration Efficiency (CE) from the Node Learning paper (Sec 3.5)

"""
import copy 
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"System Check: Running on {DEVICE}")

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
EMBEDDING_DIM = 128
NUM_CLASSES = 43
CONFIDENCE_THRESHOLD = 0.6
FUSION_EPOCHS = 20
MODEL_DIR = "./models"
DATA_DIR = "./GTSRB_data"
IMG_SIZE = 48
NUM_NODES = 2

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
            "night_cam": {"size": 48, "transform": NodeTransform(night_camera, "pole_fixed", 48)},
            "cheap_sensor": {"size": 64, "transform": NodeTransform(cheap_sensor, "handheld", 64)},
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


class FuseEncode(nn.Module):
    """
    Information fusion encoder for collaborative inference.
    Returns a fused embedding (not logits) - the node's classifier is used after.
    """
    def __init__(self, embedding_dim=EMBEDDING_DIM, n_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()

        self.emb_pad = 160 # padding for GPU optimisation 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_pad,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.compress = nn.Linear(in_features=NUM_NODES*self.emb_pad, out_features=embedding_dim)
        

    def forward(self, embeddings, padding_mask=None):
        """
        Args:
            embeddings (tagged): (batch, N_nodes, embed_dim+1) - sorted, stacked, normed and tagged. 
            padding_mask: (batch, N_nodes) - True for dropped/unavailable nodes

        Returns:
            fused: (batch, embed_dim) - fused embedding
        """
        embeddings = F.pad(embeddings, (0, self.emb_pad - embeddings.shape[-1]))
        refined = self.transformer(embeddings, src_key_padding_mask=padding_mask)

        refined = torch.flatten(refined, start_dim=1)

        fused = self.compress(refined)

        return fused
    
class FuseDecode(nn.Module):
    
    """
    Decodes the fused embedding back to normalised stacked embeddings. 
    This is only used for training purpose (forward differential loss).

    """
    def __init__(self, embedding_dim=EMBEDDING_DIM, n_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim 
        self.emb_pad = 160 # padding for GPU optimisation 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_pad,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.project = nn.Linear(in_features=embedding_dim, out_features=NUM_NODES*self.emb_pad)
        

    def forward(self, embedding, padding_mask=None):
        """
        Args:
            embedding: (batch, embed_dim) - fused embeddings by FusedEncode

        Returns:
            decoded: (batch, N_nodes, embed_dim) - decoded stacked embedding
            tags: (batch, N_nodes) - recovered availability tag
        """
        batch_size = embedding.shape[0]
        projected = self.project(embedding)
        shape_recovered = torch.reshape(projected, (batch_size, NUM_NODES, self.emb_pad))
        decoded = self.transformer(shape_recovered)

        tags = decoded[:,:,self.embedding_dim]
        decoded = decoded[:,:,:self.embedding_dim]

        return tags, decoded 


class NodeModel(nn.Module):
    """
    A node in the decentralized network.

    Each node has:
    - encoder: extracts embeddings from images
    - classifier: classifies embeddings (solo mode)
    - fuse_encode: combines own + peer embeddings (collaborative mode)
    """
    def __init__(self, include_fusion=True):
        super().__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()
        self.fuse_encode = FuseEncode() if include_fusion else None

    def forward(self, x):
        """Solo inference - just encode and classify."""
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits, embedding

    def fused_forward(self, own_embedding, peer_embeddings, mask):
        """
        Collaborative inference - fuse own embedding with peers and classify.

        Args:
            own_embedding: tuple (name, embedding) with embedding size = (batch, embed_dim) - this node's embedding
            peer_embeddings: list of tuples (name, embedding) with size = (batch, embed_dim)
            mask: (batch, NUM_NODES) - concacenated availablity tag: This is now NOT an optional parameter 

        Returns:
            logits: (batch, num_classes) 
            log_logits: (batch, num_classes), used for forward and backward loss training (symmetric KL-divergence)
        """
        if self.fuse_encode is None: 
            raise ValueError("This node has no fusion head. Initialize with include_fusion=True")

        # Sort the nodes by alphabetical order 
        all_embs_labelled = [own_embedding] + list(peer_embeddings)
        sorted_embs_labelled = sorted(all_embs_labelled, key=lambda x: x[0])
        embs_only = [item[1] for item in sorted_embs_labelled]
        
        # Stack
        stacked = torch.stack(embs_only, dim=1)  # (batch, N_nodes, embed_dim)

        mask = mask.unsqueeze(-1)
        stacked_n = nn.functional.normalize(stacked, p=2, dim=-1)
        stacked_masked_n = stacked_n * mask
        

        stacked_masked_n_tagged = torch.cat([stacked_masked_n, mask], dim=-1)

        # Create padding mask for transformer (True = ignore)
        padding_mask = (mask.squeeze(-1) == 0)

        fused = self.fuse_encode(stacked_masked_n_tagged, padding_mask)
        logits = self.classifier(fused)

        return logits 


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
# Training - Individual Node Inference Network 
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
# Training — Collaborative Inference Network (FuseEncode)
# ============================================================================

def train_fusion(nodes, multi_res_loader, node_names, p_drop=0.0, nodes_full=None, train_lambda=False):
    """
    Train the cross-attention TATE fusion head with optional node dropout.

    During training, randomly zeros out node embeddings with probability p_drop.
    This teaches the fusion head to handle missing peers gracefully.

    Args:
        nodes: Dict of trained NodeModel instances (nodes[name] = NodeModel (at least have trained solo encoder + classifier)) 
        nodes_full : Dict of trained NodeModel instances (nodes[name] = NodeModel (at least have trained solo encoder + classifier + fusion model trained with full modality)
        multi_res_loader: DataLoader returning (images_dict, labels)
        node_names: List of node names
        p_drop: Probability of dropping each node per sample (0.0 = no dropout)
    """
    # Freeze encoders, onlly train fusion heads 
    for node in nodes.values():
        node.encoder.eval()
        for param in node.encoder.parameters():
            param.requires_grad = False

    for requesting_name in node_names:
        print(f"\n  Training fusion head for: {requesting_name}")
        requesting_node = nodes[requesting_name]

        fusion_model = requesting_node.fuse_encode # student encoder model 

        # decoder is not included in NodeModel as it is not requied for inference
        # hence a new instance is generated for every training session. 
        decoder = FuseDecode() 

        fusion_model.to(DEVICE)
        decoder.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        n_nodes = len(node_names)
        
        if p_drop != 0.0:
            try: #get pre-trained encoder (teacher model)
                encoder_full = nodes_full[requesting_name]
                encoder_full.to(DEVICE)
                encoder_full.eval()
            except KeyError:
                print("Pre-trained Encoder not available.")

        # Initialise loss weighting parameters
        if train_lambda:
            lamb_1 = nn.Parameter(torch.zeros(1, device=DEVICE))
            lamb_2 = nn.Parameter(torch.zeros(1, device=DEVICE))
            lamb_3 = nn.Parameter(torch.zeros(1, device=DEVICE))
            lambdas = [lamb_1, lamb_2, lamb_3]
            full_parameters = (list(fusion_model.parameters()) + list(decoder.parameters()) + lambdas)
        else: # weights of 0.1 follows the original TATE author's implementation
            lamb_1 = 0.1
            lamb_2 = 0.1
            lamb_3 = 0.1
            full_parameters = (list(fusion_model.parameters()) + list(decoder.parameters()))

        optimizer = optim.Adam(full_parameters, lr=LR)

        epoch_bar = tqdm(range(FUSION_EPOCHS), desc="  fusion+dropout", unit="ep")
        for epoch in epoch_bar:
            fusion_model.train()
            decoder.train()
            total_loss, correct, total = 0, 0, 0

            for images_dict, labels in multi_res_loader:
                batch_size = labels.size(0)
                labels = labels.to(DEVICE)

                # Extract embeddings from each node's encoder
                emb_list = []
                node_names_sorted = sorted(node_names)
                for name in node_names_sorted:
                    images = images_dict[name].to(DEVICE)
                    with torch.no_grad():
                        emb = nodes[name].encoder(images)
                    emb_list.append((name, emb))
                
                if p_drop != 0.0:
                    emb_only = [e for _, e in emb_list]
                    stacked_full = torch.stack(emb_only, dim=1) # Full node information copy for training purpose. 

                emb_only = [e for _, e in emb_list]
                stacked = torch.stack(emb_only, dim=1)  # (batch, N, embed_dim)
                 

                mask = torch.ones(batch_size, n_nodes, 1, device=DEVICE).float()

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
                    

                    # Create padding mask for transformer (True = ignore)
                    padding_mask = (mask.squeeze(-1) == 0)
                else:
                    padding_mask = None

                stacked_n = nn.functional.normalize(stacked, p=2, dim=-1)
                stacked_masked_n = stacked_n * mask

                
                stacked_masked_n_tagged = torch.cat([stacked_masked_n, mask], dim=-1)

                # Forward through FusionEncode
                fused = fusion_model(stacked_masked_n_tagged, padding_mask)
                logits = requesting_node.classifier(fused)

                clf_error = criterion(logits, labels)

                # Forward Differential Loss if training incomplete modality model. 
                if p_drop == 0.0:
                    fwd_loss = torch.tensor(0.0, device=DEVICE)
                else:
                    stacked_full = torch.stack(emb_only, dim=1) 
                    # Attach availability tag to stacked embedding with full node info. 
                    mask_full = torch.ones_like(mask)
                    stacked_full_n = nn.functional.normalize(stacked_full, p=2, dim=-1)
                    stacked_full_n_tagged = torch.cat([stacked_full_n, mask_full], dim=-1)

                    # Passing embedding with all node information through "pre-trained encoder"
                    with torch.no_grad():
                        embed_full = nodes_full[requesting_name].fuse_encode(stacked_full_n_tagged, padding_mask=None)

                    # Forward Differential Loss (symmetric KL-divergence)
                    p_embed_full = F.softmax(embed_full, dim=-1)
                    p_fused = F.softmax(fused, dim=-1)

                    log_p_embed_full = F.log_softmax(embed_full, dim=-1)
                    log_p_fused = F.log_softmax(fused, dim=-1)
                    
                    kl1_f = F.kl_div(log_p_embed_full, p_fused, reduction='batchmean')
                    kl2_f = F.kl_div(log_p_fused, p_embed_full, reduction='batchmean') 

                    fwd_loss = kl1_f + kl2_f 
                
                # Backward Reconstruction Loss (symmetric KL-divergence).
                tags, decoded = decoder.forward(fused) # decoded = (batch, num_nodes, embed_dim)

                p_decoded = F.softmax(decoded, dim=-1)
                p_original = F.softmax(stacked_masked_n, dim=-1)

                log_p_decoded = F.log_softmax(decoded, dim=-1)
                log_p_original = F.log_softmax(stacked_masked_n, dim=-1)

                kl1_b = F.kl_div(log_p_decoded, p_original, reduction='batchmean')
                kl2_b = F.kl_div(log_p_original, p_decoded, reduction='batchmean') 
            
                bwd_loss = kl1_b + kl2_b

                # Tag Reconstruction Loss (MSE/ L1 loss)
                tags = torch.sigmoid(tags) # (batch, num_nodes)
                tag_loss = torch.nn.functional.l1_loss(mask.squeeze(-1), tags)

                # Sum up loss contributions.
                if train_lambda:
                    loss = clf_error + torch.exp(lamb_1)*fwd_loss + torch.exp(lamb_2)*bwd_loss + torch.exp(lamb_3)*tag_loss
                else: 
                    loss = clf_error + lamb_1*fwd_loss + lamb_2*bwd_loss + lamb_3*tag_loss

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
    
    return nodes



# ============================================================================
# Collaborative Inference — cross-attention fusion
# ============================================================================
@torch.no_grad()
def collaborative_inference_with_dropout(
    nodes,
    multi_res_loader,
    merge_operator=None,
    p_drop=0.0,
    confidence_threshold=CONFIDENCE_THRESHOLD,
):
    """
    Collaborative inference using each node's own fusion head.

    Each node uses its own fusion head to combine its embedding with peers.
    This is the decentralized approach where nodes don't share a fusion model.

    Args:
        nodes: Dict of NodeModel instances (with fuse_encode)
        multi_res_loader: DataLoader returning (images_dict, labels)
        merge_operator: Optional MergeOperator (overrides node's fuse_encode)
        p_drop: Probability of each peer being unavailable (0.0 = all available)
        confidence_threshold: Threshold for triggering collaboration

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
        "confidences_when_triggered": [],
        "confidences_when_not_triggered": []
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
                s['confidences_when_not_triggered'].append(confidences[name])
                pred = solo_pred
                if pred == label:
                    s["no_collab_correct"] += 1
                else:
                    s["no_collab_wrong"] += 1
            else:
                s["collab_triggered"] += 1
                s['confidences_when_triggered'].append(confidences[name])

                # Simulate node dropout for peers
                available_peers = []
                peer_confs = []

                # Initiation of generation of availability tag
                availability_dict = {n: 0.0 for n in node_names}
                availability_dict[name] = 1.0

                for peer_name in node_names:
                    if peer_name == name:
                        continue
                    # Each peer has p_drop chance of being unavailable
                    if p_drop > 0 and np.random.random() < p_drop:
                        s["peers_dropped"] += 1
                        continue
                    available_peers.append(peer_name)
                    peer_confs.append(confidences[peer_name])
                    availability_dict[peer_name] = 1.0

                node_names_sorted = sorted(node_names)
                availability_tag = torch.tensor([[availability_dict[n] for n in node_names_sorted]], device = DEVICE, dtype=torch.float32)

                if len(available_peers) == 0:
                    # No peers available, fall back to solo
                    s["no_peers_available"] += 1
                    pred = solo_pred
                else:
                    # Use requesting node's own fusion head
                    requesting_node = nodes[name]
                    requesting_emb = all_embeddings[name][i].unsqueeze(0).to(DEVICE)

                    if merge_operator is not None:
                        # Use external merge operator
                        peer_embs = [all_embeddings[p][i].unsqueeze(0).to(DEVICE) for p in available_peers]
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
                        peers = []

                        for p in node_names_sorted:
                            if p == name:
                                requester = (p, requesting_emb)
                            elif p in available_peers:
                                emb = all_embeddings[p][i].unsqueeze(0).to(DEVICE)
                                peers.append((p, emb))
                            elif p not in available_peers:
                                peers.append((p, torch.zeros_like(requesting_emb)))

                        merged_logits = requesting_node.fused_forward(
                            requester, peers, availability_tag
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
        "ir": {"distortion_fn": grayscale_ir, "preset": "pole_fixed", "size": 48},
        "colour_shifted": {"distortion_fn": colour_shifted, "preset": "vehicle_mounted", "size": 48}
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

    nodes_full = {} # {node_name: NodeModel instant (trained with full modality)}
    nodes = {} # {node_name: NodeModel instant (the most developed version)}
    any_needs_fusion_training_drop = False
    any_needs_fusion_training_full = False

    for name in NODE_CONFIGS:
        model_path_drop = os.path.join(MODEL_DIR, f"gtsrb_{name}_fusion_drop.pt") # "advanced" model trained with random dropout (P(dropout) > 0.8)
        model_path_full = os.path.join(MODEL_DIR, f"gtsrb_{name}_fusion_full.pt") # model trained with full modality, also act as pre-trained encoder
        old_model_path = os.path.join(MODEL_DIR, f"gtsrb_{name}.pt") # model with solo encoder + classifier trained only

        model = NodeModel(include_fusion=True)

        if os.path.exists(model_path_drop):
            print(f"\nLoading saved model (with fusion & trained WITH dropout): {name}")
            model.load_state_dict(torch.load(model_path_drop, map_location=DEVICE))
            model.to(DEVICE)
            nodes[name] = model
        else:
            any_needs_fusion_training_drop = True

            if os.path.exists(old_model_path):
                # Load old model (encoder + classifier only), fusion head needs training
                print(f"\nLoading saved model (no fusion): {name}")
                old_state = torch.load(old_model_path, map_location=DEVICE)
                # Filter out fuse_encode keys that don't exist in old model
                model_state = model.state_dict()
                for key in old_state:
                    if key in model_state:
                        model_state[key] = old_state[key]
                model.load_state_dict(model_state)
                nodes[name] = model.to(DEVICE)
            else:
                print(f"\nTraining node: {name}")
                model = train_node(model, train_loaders[name], name) # solo inference classifier
                any_needs_fusion_training_full = True
                nodes[name] = model

        if os.path.exists(model_path_full):
            print(f"\nLoading saved model (with fusion & trained WITHOUT dropout): {name}")
            f_model = NodeModel(include_fusion=True)
            f_model.load_state_dict(torch.load(model_path_full, map_location=DEVICE))
            nodes_full[name] = f_model.to(DEVICE)
        else:
            any_needs_fusion_training_full = True
            any_needs_fusion_training_drop = True
        
    if any_needs_fusion_training_full:
        any_needs_fusion_training_drop = True

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
        # Handle old checkpoints that lack fuse_encode keys
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
        model.eval()
        loader = test_loaders.get(name, clean_test_loader)
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits, _ = model(images)
                correct += (logits.argmax(1) == labels).sum().item()
                total += images.size(0)
        print(f"  {name:15s}  Accuracy: {correct/total*100:.1f}%")

    # --- Train each node's fusion head (if needed) ---
    if any_needs_fusion_training_drop:
        if any_needs_fusion_training_full:
            print("\nTraining per-node fusion heads (with full modality):")
            nodes_copy = {k: copy.deepcopy(v) for k, v in nodes.items()}
            nodes_full = train_fusion(nodes_copy, multi_res_train_loader, node_names, p_drop=0.0)
            for name in node_names:
                model_path = os.path.join(MODEL_DIR, f"gtsrb_{name}_fusion_full.pt")
                torch.save(nodes_full[name].state_dict(), model_path)
                print(f"  Saved {name} to {model_path}")

        print("\nTraining per-node fusion heads (without full modality):")
        nodes_drop = train_fusion(nodes, multi_res_train_loader, node_names, p_drop=0.8, nodes_full=nodes_full)
        for name in node_names:
            model_path = os.path.join(MODEL_DIR, f"gtsrb_{name}_fusion_drop.pt")
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

    fusion_stats = {name: {"correct": 0, "total": 0} for name in node_names}

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
                own_emb = (name, embeddings[name])
                peer_embs = [(p, embeddings[p]) for p in node_names if p != name]
                mask_eval = torch.ones((batch_size, len(node_names)), device=DEVICE)

                logits = nodes[name].fused_forward(own_emb, peer_embs, mask_eval)
                fusion_stats[name]["correct"] += (logits.argmax(1) == labels).sum().item()
                fusion_stats[name]["total"] += batch_size

    print("\n  Per-node fusion accuracy (using own fusion head):")
    for name in node_names:
        acc = fusion_stats[name]["correct"] / fusion_stats[name]["total"] * 100
        print(f"    {name:15s}  Accuracy: {acc:.1f}% ]")

    # --- Collaborative inference with threshold ---
    print("\n" + "=" * 60)
    print(f"COLLABORATIVE INFERENCE (threshold={CONFIDENCE_THRESHOLD})")
    print(f"Confidence measure: normalized entropy")
    print(f"Fusion method: per-node fusion heads (decentralized)")
    print("=" * 60)

    stats, n_samples = collaborative_inference_with_dropout(nodes, multi_res_test_loader, p_drop=0.0)

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
