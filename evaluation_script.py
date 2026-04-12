from pathlib import Path
import shutil
import torchvision
import requests
import os
import random
import csv
import pandas as pd

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
TEST_CSV = "./GTSRB_data/Test.csv"


def download_gtsrb_dataset():
    print("Step 1: Downloading GTSRB dataset...")
    # This downloads the raw data into a temporary 'temp_gtsrb' folder
    train_data = torchvision.datasets.GTSRB(root='temp_gtsrb', split='train', download=True)
    test_data = torchvision.datasets.GTSRB(root='temp_gtsrb', split='test', download=True)

    # The actual images are nested deep inside 'temp_gtsrb/gtsrb/GTSRB/Training'
    # We want to move them to a clean 'GTSRB_torchvision' folder
    source_dir = os.path.join('temp_gtsrb', 'gtsrb', 'GTSRB', 'Training')
    target_dir = 'GTSRB_torchvision'

    print(f"Step 2: Reorganizing data into {target_dir}...")
    if os.path.exists(source_dir):
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.move(source_dir, target_dir)
        
        # Cleanup temporary files
        shutil.rmtree('temp_gtsrb')
        print("Success! Data is ready in GTSRB_torchvision/.")
    else:
        print("Error: Could not find the downloaded source folder.")


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
            continue

        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".ppm")])
        random.shuffle(files)

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

    # Write CSVs
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


def classify_image(image_path: Path, method: str = "collaborative") -> dict:
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/classify",
            files={"file": (image_path.name, f, "image/x-portable-pixmap")},
            params={"method": method},
        )
    response.raise_for_status()
    return response.json()


def main():

    test_count = 0
    solo_correct_count = 0
    fusion_correct_count = 0
    collaborative_correct_count = 0
    collaborative_fusion_trigger_count = 0
    collaborative_fusion_fixed_count = 0
    collaborative_fusion_broke_count = 0

    for _, row in pd.read_csv(TEST_CSV).iterrows():
        image_path = Path(row["Path"])
        label = row["ClassId"]

        solo_output = classify_image(image_path, "solo")
        fusion_output = classify_image(image_path, "fusion")
        collaborative_output = classify_image(image_path, "collaborative")

        test_count += 1
        if solo_output['pred_class'] == label:
            solo_correct_count += 1
        if fusion_output['pred_class'] == label:
            fusion_correct_count += 1
        if collaborative_output['pred_class'] == label:
            collaborative_correct_count += 1
        if collaborative_output['method'] == 'fusion_inference':
            collaborative_fusion_trigger_count += 1

            if collaborative_output['extra_info']['solo_pred_class'] != label and collaborative_output['pred_class'] == label:
                collaborative_fusion_fixed_count += 1
            if collaborative_output['extra_info']['solo_pred_class'] == label and collaborative_output['pred_class'] != label:
                collaborative_fusion_broke_count += 1

        print(
            f"[{test_count}] "
            f"Solo: {solo_correct_count}/{test_count} ({100*solo_correct_count/test_count:.1f}%)  "
            f"Fusion: {fusion_correct_count}/{test_count} ({100*fusion_correct_count/test_count:.1f}%)  "
            f"Collaborative: {collaborative_correct_count}/{test_count} ({100*collaborative_correct_count/test_count:.1f}%)  "
            f"| Fusion triggered: {collaborative_fusion_trigger_count}  "
            f"Fixed: {collaborative_fusion_fixed_count}  "
            f"Broke: {collaborative_fusion_broke_count}"
        )

if __name__ == "__main__":
    download_gtsrb_dataset()
    prepare_gtsrb_dataset()
    main()
