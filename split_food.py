import os
import shutil
from PIL import Image

# === Configuration ===
raw_images_root = "/home/classes/ee7722/ee772210/Downloads/food-101/images/"  # path to original class folders (e.g. apple_pie/*.jpg)
output_root = "datasets/food101_yolo"
train_txt = "/home/classes/ee7722/ee772210/Downloads/food-101/meta/train.txt"
val_txt = "/home/classes/ee7722/ee772210/Downloads/food-101/meta/test.txt"  # Food-101 calls it test.txt

# === Output directories ===
image_train = os.path.join(output_root, "images", "train")
image_val = os.path.join(output_root, "images", "val")
label_train = os.path.join(output_root, "labels", "train")
label_val = os.path.join(output_root, "labels", "val")

# === Create dirs if they don't exist ===
for d in [image_train, image_val, label_train, label_val]:
    os.makedirs(d, exist_ok=True)

# === Load class names ===
with open("/home/classes/ee7722/ee772210/Downloads/food-101/meta/classes.txt") as f:
    class_names = [line.strip() for line in f]
class_dict = {name: idx for idx, name in enumerate(class_names)}

# === Function to convert image and generate label ===
def process_list(txt_file, image_dir, label_dir):
    with open(txt_file, "r") as f:
        for line in f:
            rel_path = line.strip()  # e.g., apple_pie/123456
            class_name = rel_path.split("/")[0]
            img_name = rel_path.split("/")[1] + ".jpg"
            class_id = class_dict[class_name]

            src_img_path = os.path.join(raw_images_root, class_name, img_name)
            dst_img_path = os.path.join(image_dir, f"{class_name}_{img_name}")
            dst_lbl_path = os.path.join(label_dir, f"{class_name}_{img_name.replace('.jpg', '.txt')}")

            if not os.path.exists(src_img_path):
                print(f"Image missing: {src_img_path}")
                continue

            shutil.copy2(src_img_path, dst_img_path)

            # Create a bounding box that spans most of the image (assume object-centered)
            try:
                with Image.open(src_img_path) as img:
                    w, h = img.size
                # YOLO format: <class> <x_center> <y_center> <width> <height> (normalized)
                bbox = [class_id, 0.5, 0.5, 0.9, 0.9]
                with open(dst_lbl_path, "w") as f_lbl:
                    f_lbl.write(" ".join([str(x) for x in bbox]) + "\n")
            except Exception as e:
                print(f"Failed processing {src_img_path}: {e}")

# === Generate train and val splits with labels ===
process_list(train_txt, image_train, label_train)
process_list(val_txt, image_val, label_val)

print("✅ Dataset organized and YOLO labels generated.")