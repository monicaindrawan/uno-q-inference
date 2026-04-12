"""
Train 2 Nodes: IR + Colour-Shifted
====================================

Trains only 2 camera nodes (grayscale/IR and colour-shifted) to collaborate.
Fusion heads are trained with peer dropout that can drop ALL peers (no min-cap),
so each node learns to fall back on its own embedding when the peer is unavailable.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image

from distortions_final import (
    grayscale_ir,
    colour_shifted,
    NodeTransform,
    RandomTransformChain,
    stereo_shift,
    horizontal_sign_angle,
    vertical_sign_angle,
    sign_approaching,
    overcast_flat_light,
    random_occlusion,
)
from node_model import NodeModel

BATCH_SIZE = 64
EMBEDDING_DIM = 128
NUM_CLASSES = 43
CONFIDENCE_THRESHOLD = 0.6
MODEL_DIR = "./models"
DATA_DIR = "./GTSRB_data"
IMG_SIZE = 48
DEVICE = "cpu"

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
# Confidence measure — normalized entropy
# ============================================================================
def confidence_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=-1)
    max_entropy = np.log(probs.size(-1))
    return 1.0 - entropy / max_entropy

confidence_fn = confidence_entropy

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


# --- 2-node configuration ---
NODE_CONFIGS = {
    "ir": {"distortion_fn": grayscale_ir, "preset": "pole_fixed", "size": 48},
    "colour_shifted": {"distortion_fn": colour_shifted, "preset": "vehicle_mounted", "size": 48},
}

RANDOM_TRANSFORMS = {
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
}

def main():
    t_start = time.time()
    test_csv = os.path.join(DATA_DIR, "Test.csv")
    node_names = list(NODE_CONFIGS.keys())

    # --- Create datasets and loaders ---
    test_loaders = {}
    for name, config in NODE_CONFIGS.items():
        random_chain = RandomTransformChain(RANDOM_TRANSFORMS.get(name, []))
        test_transform = NodeTransform(
            distortion_fn=config["distortion_fn"],
            preset=config["preset"],
            output_size=config["size"],
            training=False,
            random_transforms=random_chain,
        )
        test_ds = DistortedGTSRB(test_csv, DATA_DIR, transform=test_transform)

        test_loaders[name] = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

    print(f"Nodes: {', '.join(node_names)}, Classes: {NUM_CLASSES}")

    # --- Multi-resolution datasets for fusion training/inference ---
    def build_multi_res_configs(training):
        result = {}
        for name, cfg in NODE_CONFIGS.items():
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

    multi_res_test_ds = MultiResolutionDataset(
        test_csv, DATA_DIR, build_multi_res_configs(training=False)
    )
    multi_res_test_loader = DataLoader(
        multi_res_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # --- Train or load node encoders ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    nodes = {}
    any_needs_fusion_training = False

    for name in node_names:
        model_path = os.path.join(MODEL_DIR, f"gtsrb_{name}_2node_with_fusion.pt")

        model = NodeModel(include_fusion=True)

        print(f"\nLoading saved 2-node model (with fusion): {name}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)

        nodes[name] = model

    # --- Evaluate solo performance ---
    print("\n" + "=" * 60)
    print("SOLO PERFORMANCE (each node classifies alone)")
    print("=" * 60)

    for name, model in nodes.items():
        model.eval()
        loader = test_loaders[name]
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits, _ = model(images)
                correct += (logits.argmax(1) == labels).sum().item()
                total += images.size(0)
        print(f"  {name:15s}  Accuracy: {correct/total*100:.1f}%")

    # --- Evaluate per-node fusion (always fuse, all peers available) ---
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

            embeddings = {}
            for name in node_names:
                images = images_dict[name].to(DEVICE)
                embeddings[name] = nodes[name].encoder(images)

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
        peer_names = [p for p in node_names if p != name]
        print(f"    {name:15s}  Accuracy: {acc:.1f}%  "
              f"Attention: [self={avg_weights[0]:.3f}, "
              f"{peer_names[0]}={avg_weights[1]:.3f}]")

    # --- Collaborative inference (no dropout baseline) ---
    print("\n" + "=" * 60)
    print(f"COLLABORATIVE INFERENCE — NO DROPOUT (threshold={CONFIDENCE_THRESHOLD})")
    print(f"Confidence measure: normalized entropy")
    print(f"Fusion method: per-node fusion heads (decentralized, 2-node)")
    print("=" * 60)

    stats, n_samples = collaborative_inference_with_dropout(
        nodes, multi_res_test_loader, p_drop=0.0,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    # --- Collaborative inference WITH dropout ---
    print("\n" + "=" * 60)
    print(f"COLLABORATIVE INFERENCE — p_drop=0.3 (threshold={CONFIDENCE_THRESHOLD})")
    print(f"Peer can be completely unavailable (no min-cap)")
    print("=" * 60)

    stats_drop, n_samples_drop = collaborative_inference_with_dropout(
        nodes, multi_res_test_loader, p_drop=0.3,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    # --- Collaboration Efficiency ---
    print("\n" + "=" * 60)
    print("COLLABORATION EFFICIENCY (CE = accuracy gain / bytes exchanged)")
    print("=" * 60)
    embedding_bytes = EMBEDDING_DIM * 4  # sizeof(float32)
    n_peers = len(node_names) - 1
    for label, run_stats, run_n in [
        ("No dropout", stats, n_samples),
        ("p_drop=0.3", stats_drop, n_samples_drop),
    ]:
        print(f"\n  [{label}]")
        for name in node_names:
            s = run_stats[name]
            triggered = s["collab_triggered"]
            no_peers = s.get("no_peers_available", 0)
            actual_comms = triggered - no_peers
            total_bytes = actual_comms * embedding_bytes * n_peers
            gain = s["collab_fixed"] - s["collab_broke"]
            ce = gain / max(total_bytes, 1)
            print(f"    {name:15s}  CE: {ce:.6f} correct_samples/byte  "
                  f"(total bytes exchanged: {total_bytes:,})")

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed time: {int(elapsed//60)}m {int(elapsed%60)}s")


if __name__ == "__main__":
    main()
