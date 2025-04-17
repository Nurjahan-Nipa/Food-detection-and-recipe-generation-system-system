import os
import shutil

# Configuration
raw_images_root = "/Users/epic/Downloads/ML/food-101/images/"


# === CONFIGURATION ===
#raw_images_root = "original_data/images"
output_root = "datasets/food101_yolo/images"
train_txt = "/Users/epic/Downloads/ML/food-101/meta/train.txt"
test_txt = "/Users/epic/Downloads/ML/food-101/meta/test.txt"

train_out = os.path.join(output_root, "train")
val_out = os.path.join(output_root, "val")

os.makedirs(train_out, exist_ok=True)
os.makedirs(val_out, exist_ok=True)

def copy_images(txt_file, dest_dir):
    with open(txt_file, "r") as f:
        for line in f:
            rel_path = line.strip()
            src_path = os.path.join(raw_images_root, rel_path + ".jpg")
            dest_path = os.path.join(dest_dir, rel_path.replace("/", "_") + ".jpg")
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
            else:
                print(f"Missing image: {src_path}")

copy_images(train_txt, train_out)
copy_images(test_txt, val_out)

print("✅ Image splitting complete.")
