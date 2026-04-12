"""
Inference Only: 2 Nodes (IR + Colour-Shifted)
===============================================

Loads pre-trained .pth weights for both node encoders and fusion heads,
then runs the full evaluation suite (solo, per-node fusion, collaborative).

Expected weight files (in MODEL_DIR):
  gtsrb_ir_2node_with_fusion.pt
  gtsrb_colour_shifted_2node_with_fusion.pt
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

from node_learning_gtsrb import (
    NodeModel,
    DistortedGTSRB,
    MultiResolutionDataset,
    collaborative_inference_with_dropout,
    DEVICE,
    BATCH_SIZE,
    EMBEDDING_DIM,
    NUM_CLASSES,
    CONFIDENCE_THRESHOLD,
    MODEL_DIR,
    DATA_DIR,
)
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

# --- 2-node configuration (must match training) ---
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


def load_nodes(node_names, model_dir=MODEL_DIR):
    """Load all node models from saved .pth weights. Exits if any are missing."""
    nodes = {}
    for name in node_names:
        model_path = os.path.join(model_dir, f"gtsrb_{name}_2node_with_fusion.pt")
        if not os.path.exists(model_path):
            print(f"ERROR: Missing weights for '{name}': {model_path}")
            print("Run train_2_models.py first to generate the weights.")
            sys.exit(1)

        model = NodeModel(include_fusion=True)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"Loaded: {model_path}")
        nodes[name] = model

    return nodes


def build_test_loaders(node_names, test_csv):
    """Build per-node single-image test loaders."""
    loaders = {}
    for name in node_names:
        cfg = NODE_CONFIGS[name]
        random_chain = RandomTransformChain(RANDOM_TRANSFORMS.get(name, []))
        transform = NodeTransform(
            distortion_fn=cfg["distortion_fn"],
            preset=cfg["preset"],
            output_size=cfg["size"],
            training=False,
            random_transforms=random_chain,
        )
        ds = DistortedGTSRB(test_csv, DATA_DIR, transform=transform)
        loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return loaders


def build_multi_res_loader(node_names, test_csv):
    """Build a MultiResolutionDataset loader for fusion/collab inference."""
    configs = {}
    for name in node_names:
        cfg = NODE_CONFIGS[name]
        random_chain = RandomTransformChain(RANDOM_TRANSFORMS.get(name, []))
        transform = NodeTransform(
            distortion_fn=cfg["distortion_fn"],
            preset=cfg["preset"],
            output_size=cfg["size"],
            training=False,
            random_transforms=random_chain,
        )
        configs[name] = {"size": cfg["size"], "transform": transform}

    ds = MultiResolutionDataset(test_csv, DATA_DIR, configs)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0), len(ds)


def eval_solo(nodes, test_loaders, node_names):
    print("\n" + "=" * 60)
    print("SOLO PERFORMANCE (each node classifies alone)")
    print("=" * 60)
    for name in node_names:
        model = nodes[name]
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loaders[name]:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits, _ = model(images)
                correct += (logits.argmax(1) == labels).sum().item()
                total += images.size(0)
        print(f"  {name:15s}  Accuracy: {correct/total*100:.1f}%  ({correct}/{total})")


def eval_fusion(nodes, multi_res_loader, node_names):
    print("\n" + "=" * 60)
    print("PER-NODE FUSION PERFORMANCE (all peers available)")
    print("=" * 60)

    stats = {name: {"correct": 0, "total": 0, "attn_weights": []} for name in node_names}

    with torch.no_grad():
        for images_dict, labels in multi_res_loader:
            labels = labels.to(DEVICE)
            batch_size = labels.size(0)

            embeddings = {
                name: nodes[name].encoder(images_dict[name].to(DEVICE))
                for name in node_names
            }

            for name in node_names:
                own_emb = embeddings[name]
                peer_embs = [embeddings[p] for p in node_names if p != name]
                logits, weights = nodes[name].fused_forward(own_emb, peer_embs)
                stats[name]["correct"] += (logits.argmax(1) == labels).sum().item()
                stats[name]["total"] += batch_size
                stats[name]["attn_weights"].append(weights.cpu())

    for name in node_names:
        acc = stats[name]["correct"] / stats[name]["total"] * 100
        avg_w = torch.cat(stats[name]["attn_weights"]).mean(dim=0)
        peers = [p for p in node_names if p != name]
        print(f"  {name:15s}  Accuracy: {acc:.1f}%  "
              f"Attention: [self={avg_w[0]:.3f}, {peers[0]}={avg_w[1]:.3f}]")


def print_collab_stats(stats, n_samples, node_names):
    for name in node_names:
        s = stats[name]
        solo_acc   = s["solo_correct"]   / n_samples * 100
        collab_acc = s["collab_correct"] / n_samples * 100
        triggered  = s["collab_triggered"]
        delta      = collab_acc - solo_acc

        print(f"\n  --- {name} ---")
        print(f"  Solo accuracy:    {solo_acc:.1f}%")
        print(f"  Collab accuracy:  {collab_acc:.1f}%  (Δ {delta:+.1f}%)")
        print(f"  Communication triggered: {triggered}/{n_samples} "
              f"({triggered/n_samples*100:.1f}%)")

        if triggered > 0:
            no_peers = s.get("no_peers_available", 0)
            if no_peers > 0:
                print(f"  Peer unavailable (fell back to solo): {no_peers}/{triggered} "
                      f"({no_peers/triggered*100:.1f}%)")

            print(f"\n  When collaboration WAS triggered ({triggered} samples):")
            print(f"    Solo wrong → collab FIXED it:     {s['collab_fixed']:4d}  "
                  f"({s['collab_fixed']/triggered*100:5.1f}%)")
            print(f"    Solo wrong → collab STAYED wrong: {s['collab_stayed_wrong']:4d}  "
                  f"({s['collab_stayed_wrong']/triggered*100:5.1f}%)")
            print(f"    Solo right → collab KEPT it:      {s['collab_stayed_right']:4d}  "
                  f"({s['collab_stayed_right']/triggered*100:5.1f}%)")
            print(f"    Solo right → collab BROKE it:     {s['collab_broke']:4d}  "
                  f"({s['collab_broke']/triggered*100:5.1f}%)")

            net = s["collab_fixed"] - s["collab_broke"]
            print(f"    Net effect: {net:+d} ({net/triggered*100:+.1f}% of triggered)")


def eval_collaborative(nodes, multi_res_loader, node_names, n_samples):
    embedding_bytes = EMBEDDING_DIM * 4  # sizeof(float32)
    n_peers = len(node_names) - 1

    runs = [
        ("NO DROPOUT",  0.0),
        ("p_drop=0.3",  0.3),
    ]

    all_stats = {}
    for label, p_drop in runs:
        print("\n" + "=" * 60)
        print(f"COLLABORATIVE INFERENCE — {label} (threshold={CONFIDENCE_THRESHOLD})")
        print("=" * 60)

        stats, n = collaborative_inference_with_dropout(
            nodes, multi_res_loader, p_drop=p_drop,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        print_collab_stats(stats, n, node_names)
        all_stats[label] = (stats, n)

    # Collaboration efficiency
    print("\n" + "=" * 60)
    print("COLLABORATION EFFICIENCY (CE = accuracy gain / bytes exchanged)")
    print("=" * 60)
    for label, (run_stats, run_n) in all_stats.items():
        print(f"\n  [{label}]")
        for name in node_names:
            s = run_stats[name]
            triggered    = s["collab_triggered"]
            no_peers     = s.get("no_peers_available", 0)
            actual_comms = triggered - no_peers
            total_bytes  = actual_comms * embedding_bytes * n_peers
            gain         = s["collab_fixed"] - s["collab_broke"]
            ce           = gain / max(total_bytes, 1)
            print(f"    {name:15s}  CE: {ce:.6f} correct_samples/byte  "
                  f"(bytes exchanged: {total_bytes:,})")


def main():
    node_names = list(NODE_CONFIGS.keys())
    test_csv   = os.path.join(DATA_DIR, "Test.csv")

    print(f"Device: {DEVICE}")
    print(f"Nodes:  {', '.join(node_names)}")

    nodes = load_nodes(node_names)

    test_loaders = build_test_loaders(node_names, test_csv)
    multi_res_loader, n_samples = build_multi_res_loader(node_names, test_csv)

    print(f"Test set: {n_samples} images")

    eval_solo(nodes, test_loaders, node_names)
    eval_fusion(nodes, multi_res_loader, node_names)
    eval_collaborative(nodes, multi_res_loader, node_names, n_samples)


if __name__ == "__main__":
    main()
