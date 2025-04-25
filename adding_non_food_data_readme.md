# CookVision - Training Overview

## 📦 Incorporating COCO Unlabeled Dataset for Background Augmentation

To improve model robustness and reduce false positives, we enhanced the training data by integrating **non-food background images** from the **COCO Unlabeled 2017** dataset into the Food-101 training set.

### 📆 Dataset Composition
- **Primary dataset**: Food-101 (converted to object detection format)
- **Background images**: COCO Unlabeled 2017 dataset
  - Images **without bounding box labels** (treated as background)
  - Helps the model distinguish food items from unrelated objects and scenes

### 🔧 Integration Process
1. **Food-101 images and labels** were retained as-is for food detection.
2. **COCO unlabeled images** were added to the training set:
   - COCO images were placed in the `images/train/` directory.
   - Empty `.txt` label files were generated for each COCO image under `labels/train/` to indicate no object detection.
3. **Data.yaml** remained unchanged:
   - Only Food-101 food classes were listed in `names`.
   - COCO background images were not assigned any new classes.

### 🔥 Key Points
- **No changes** were made to the Food-101 class list or the `dish2ingredients.json` file.
- The model was trained to recognize Food-101 food items while ignoring non-food distractions.
- **Training stability** and **false positive suppression** were significantly improved.

> **Note:** Since no new food classes were introduced, **ingredient inference mapping (`dish2ingredients.json`) remained valid and unchanged**.

---

## 🔬 Training Details

- **Model**: YOLOv8n (Nano version)
- **Epochs**: 20
- **Input Resolution**: 640 × 640 pixels
- **Framework**: Ultralytics YOLOv8 (Python library)
- **Device**: NVIDIA GeForce RTX 4060 Laptop GPU
- **Training results directory**:

```plaintext
runs/detect/train13/
```

Contents include:
- `best.pt` — Best-performing model weights
- `results.png` — Training loss and mAP curves
- `confusion_matrix.png`, `F1_curve.png`, `PR_curve.png` — Evaluation visuals
- `opt.yaml`, `hyp.yaml` — Training configuration and hyperparameters

---

## 🚀 Next Steps

- Use `best.pt` for inference and recipe generation.
- Validate the model using:
  ```bash
  yolo detect val model=runs/detect/train13/weights/best.pt data=data.yaml imgsz=640 device=0
  ```
- Deploy in the CookVision pipeline to detect dishes, infer ingredients, and generate recipes.
