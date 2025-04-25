import os
import zipfile
import shutil
import requests
from tqdm import tqdm

# === Config ===
raw_coco_dir= "dataset"
dataset_dir = "dataset/food101_yolo"
coco_zip_url = "http://images.cocodataset.org/zips/unlabeled2017.zip"
zip_path = os.path.join(raw_coco_dir, "unlabeled2017.zip")
extract_dir = os.path.join(raw_coco_dir, "coco_unlabeled")
train_img_dir = os.path.join(dataset_dir, "images/train")
train_lbl_dir = os.path.join(dataset_dir, "labels/train")
n_images = 500  # number of background images to extract

# === Step 1: Download COCO Unlabeled Zip ===
if not os.path.exists(zip_path):
    print("📥 Downloading COCO unlabeled dataset...")
    with requests.get(coco_zip_url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
else:
    print("✅ Zip already exists, skipping download.")

# === Step 2: Extract ===
if not os.path.exists(extract_dir):
    print("📂 Extracting COCO images...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
else:
    print("✅ Extracted already.")

# === Step 3: Move subset of images into YOLO train set ===
coco_images_dir = os.path.join(extract_dir, "unlabeled2017")
coco_images = sorted([f for f in os.listdir(coco_images_dir) if f.endswith(".jpg")])[:n_images]

for i, fname in enumerate(coco_images):
    src = os.path.join(coco_images_dir, fname)
    new_name = f"bgcoco_{i:05d}.jpg"
    dst_img = os.path.join(train_img_dir, new_name)
    dst_lbl = os.path.join(train_lbl_dir, new_name.replace(".jpg", ".txt"))

    shutil.copy2(src, dst_img)
    open(dst_lbl, "w").close()  # empty label

print(f"✅ Injected {len(coco_images)} COCO background images into YOLO training set.")
