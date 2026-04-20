"""
Evaluation Script — End-to-End GTSRB Inference Benchmark
=========================================================

Runs three inference modes against a live node server and reports accuracy
and latency side-by-side for every test image:

    Solo          : only the local node's CNN classifier is used
    Fusion        : always combine peer embeddings via the fusion head
    Collaborative : use solo unless confidence is low, then trigger fusion

For Collaborative mode, the script also tracks:
    - Fusion trigger rate  : how often low confidence caused a peer call
    - Fixed count          : solo was wrong, fusion corrected it
    - Broke count          : solo was right, fusion made it wrong

Pipeline (run once on a fresh machine):
    1. download_gtsrb_dataset()  — download via torchvision, reorganise folders
    2. prepare_gtsrb_dataset()   — split into Train/Test CSVs (80/20)
    3. main()                    — iterate Test.csv, call /classify, print stats

Usage:
    BASE_URL=http://<node-ip>:8000 python evaluation_script.py
"""

from pathlib import Path
import shutil
import time
import torchvision
import requests
import os
import random
import csv
import pandas as pd

BASE_URL = os.environ.get("BASE_URL", "http://192.168.0.102:8000")
TEST_CSV = "./GTSRB_data/Test.csv"


# =============================================================================
# Dataset Download
# Downloads GTSRB via torchvision into a temp folder, then moves the image
# tree to a clean 'GTSRB_torchvision/' directory and removes the temp files.
# =============================================================================

def download_gtsrb_dataset():
    print("Step 1: Downloading GTSRB dataset...")
    # torchvision downloads raw archives into temp_gtsrb/
    train_data = torchvision.datasets.GTSRB(root='temp_gtsrb', split='train', download=True)
    test_data = torchvision.datasets.GTSRB(root='temp_gtsrb', split='test', download=True)

    # Images land deep inside the torchvision cache hierarchy; pull them out
    source_dir = os.path.join('temp_gtsrb', 'gtsrb', 'GTSRB', 'Training')
    target_dir = 'GTSRB_torchvision'

    print(f"Step 2: Reorganizing data into {target_dir}...")
    if os.path.exists(source_dir):
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.move(source_dir, target_dir)
        shutil.rmtree('temp_gtsrb')
        print("Success! Data is ready in GTSRB_torchvision/.")
    else:
        print("Error: Could not find the downloaded source folder.")


# =============================================================================
# Dataset Preparation
# Scans GTSRB_torchvision/ (one sub-folder per class, named "00000"–"00042"),
# applies an 80/20 random train/test split per class, and writes two CSVs:
#   GTSRB_data/Train.csv  — used by node_learning_gtsrb.py for training
#   GTSRB_data/Test.csv   — used by main() below for evaluation
# =============================================================================

def prepare_gtsrb_dataset():
    IMAGES_DIR = "GTSRB_torchvision"
    OUTPUT_DIR = "GTSRB_data"
    TRAIN_SPLIT = 0.8
    RANDOM_SEED = 42

    random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_rows = []
    test_rows = []

    class_folders = sorted(os.listdir(IMAGES_DIR))
    total_images = 0

    for folder in class_folders:
        folder_path = os.path.join(IMAGES_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        try:
            class_id = int(folder)
        except ValueError:
            continue  # skip non-numeric entries (e.g. .DS_Store)

        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".ppm")])
        random.shuffle(files)

        # Ensure at least one training sample even for tiny classes
        n_train = max(1, int(len(files) * TRAIN_SPLIT))
        train_files = files[:n_train]
        test_files = files[n_train:]

        for f in train_files:
            rel_path = os.path.join(folder_path, f).replace("\\", "/")
            train_rows.append({"Path": rel_path, "ClassId": class_id})

        for f in test_files:
            rel_path = os.path.join(folder_path, f).replace("\\", "/")
            test_rows.append({"Path": rel_path, "ClassId": class_id})

        total_images += len(files)

    # Write Train.csv and Test.csv
    train_csv = os.path.join(OUTPUT_DIR, "Train.csv")
    test_csv  = os.path.join(OUTPUT_DIR, "Test.csv")

    fieldnames = ["Path", "ClassId"]

    with open(train_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)

    with open(test_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)

    print(f"Total images : {total_images}")
    print(f"Train samples: {len(train_rows)}  -> {train_csv}")
    print(f"Test  samples: {len(test_rows)}  -> {test_csv}")
    print(f"Classes      : {len(class_folders)}")
    print("\nDone. You can now run: python node_learning_gtsrb.py")


# =============================================================================
# Inference Helper
# POSTs a single .ppm file to the node's /classify endpoint.
# Always uses fusion_head as the merge operator.
# =============================================================================

def classify_image(image_path: Path, method: str = "collaborative") -> dict:
    """Send one image to the running node server and return the JSON response.

    Args:
        image_path : path to a .ppm file from Test.csv
        method     : "solo" | "fusion" | "collaborative"
    """
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/classify",
            files={"file": (image_path.name, f, "image/x-portable-pixmap")},
            params={"method": method, "merge_operator": "fusion_head"
            ""},
        )
    response.raise_for_status()
    return response.json()


# =============================================================================
# Main Evaluation Loop
# Iterates every row in Test.csv, runs all three inference modes, accumulates
# accuracy and latency counters, and prints a running summary after each image.
#
# Collaborative-specific counters:
#   trigger_count — fusion was invoked (solo confidence was below threshold)
#   fixed_count   — solo was wrong AND fusion corrected it
#   broke_count   — solo was right BUT fusion introduced a mistake
# =============================================================================

def main():

    test_count = 0
    solo_correct_count = 0
    fusion_correct_count = 0
    collaborative_correct_count = 0
    collaborative_fusion_trigger_count = 0
    collaborative_fusion_fixed_count = 0
    collaborative_fusion_broke_count = 0
    solo_total_time = 0.0
    fusion_total_time = 0.0
    collaborative_total_time = 0.0

    for _, row in pd.read_csv(TEST_CSV).iterrows():
        image_path = Path(row["Path"])
        label = row["ClassId"]

        # Time each inference mode independently
        t0 = time.perf_counter(); solo_output = classify_image(image_path, "solo"); solo_total_time += time.perf_counter() - t0
        t0 = time.perf_counter(); fusion_output = classify_image(image_path, "fusion"); fusion_total_time += time.perf_counter() - t0
        t0 = time.perf_counter(); collaborative_output = classify_image(image_path, "collaborative"); collaborative_total_time += time.perf_counter() - t0

        test_count += 1
        if solo_output['pred_class'] == label:
            solo_correct_count += 1
        if fusion_output['pred_class'] == label:
            fusion_correct_count += 1
        if collaborative_output['pred_class'] == label:
            collaborative_correct_count += 1

        # Analyse when collaborative mode escalated to fusion
        if collaborative_output['method'] == 'fusion_inference':
            collaborative_fusion_trigger_count += 1

            solo_pred = collaborative_output['extra_info']['solo_pred_class']
            if solo_pred != label and collaborative_output['pred_class'] == label:
                collaborative_fusion_fixed_count += 1   # fusion rescued a wrong solo
            if solo_pred == label and collaborative_output['pred_class'] != label:
                collaborative_fusion_broke_count += 1   # fusion overrode a correct solo

        print(
            f"[{test_count}] "
            f"Solo: {solo_correct_count}/{test_count} ({100*solo_correct_count/test_count:.1f}%)  "
            f"Fusion: {fusion_correct_count}/{test_count} ({100*fusion_correct_count/test_count:.1f}%)  "
            f"Collaborative: {collaborative_correct_count}/{test_count} ({100*collaborative_correct_count/test_count:.1f}%)  "
            f"| Fusion triggered: {collaborative_fusion_trigger_count}  "
            f"Fixed: {collaborative_fusion_fixed_count}  "
            f"Broke: {collaborative_fusion_broke_count}  "
            f"| Avg ms/img — Solo: {1000*solo_total_time/test_count:.1f}  "
            f"Fusion: {1000*fusion_total_time/test_count:.1f}  "
            f"Collaborative: {1000*collaborative_total_time/test_count:.1f}"
        )

if __name__ == "__main__":
    download_gtsrb_dataset()
    prepare_gtsrb_dataset()
    main()
