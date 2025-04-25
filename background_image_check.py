import os

image_dir = "dataset/food101_yolo/images/train"
label_dir = "dataset/food101_yolo/labels/train"

failures = []

for fname in os.listdir(image_dir):
    if fname.startswith("bgcoco") and fname.endswith(".jpg"):
        label_path = os.path.join(label_dir, fname.replace(".jpg", ".txt"))
        if not os.path.exists(label_path):
            failures.append((fname, "Missing label"))
        elif os.path.getsize(label_path) != 0:
            failures.append((fname, "Label not empty"))

if not failures:
    print("✅ All background images have empty YOLO labels!")
else:
    print("❌ Issues found:")
    for f, reason in failures:
        print(f"  {f}: {reason}")
