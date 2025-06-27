import os
import yaml
import cv2
import matplotlib.pyplot as plt
from glob import glob
from ultralytics import YOLO

# === CONFIGURATION ===
pt_model_path = "C:/training/kaggle/models/wood_defect_yolov8n.pt"
dataset_yaml_path = "C:/training/kaggle/dataset.yaml"
img_dir = "C:/training/kaggle/val/images"
label_dir = "C:/training/kaggle/val/labels"

# === LOAD CLASSES ===
with open(dataset_yaml_path, "r") as f:
    data = yaml.safe_load(f)
class_names = data["names"]

# === LOAD MODEL ===
model = YOLO(pt_model_path)

# === ANNOTATION LOADER ===
def load_annotations(label_path, img_shape):
    h, w = img_shape[:2]
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x, y, bw, bh = map(float, parts)
                    x1 = int((x - bw / 2) * w)
                    y1 = int((y - bh / 2) * h)
                    x2 = int((x + bw / 2) * w)
                    y2 = int((y + bh / 2) * h)
                    boxes.append((int(cls), x1, y1, x2, y2))
    return boxes

# === DRAW BOXES ===
def draw_boxes(img, preds, gt_boxes):
    img = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Predicted boxes in green
    for box in preds.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{class_names[cls]} {conf:.2f}"
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Ground truth boxes in red
    for cls, x1, y1, x2, y2 in gt_boxes:
        label = f"{class_names[cls]} (GT)"
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return img_rgb

# === MAIN LOOP ===
image_paths = glob(os.path.join(img_dir, "*.jpg"))
for img_path in image_paths:
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_dir, base + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")
        continue

    gt_boxes = load_annotations(label_path, img.shape)
    result = model(img)[0]  # Only the first image result
    output_img = draw_boxes(img, result, gt_boxes)

    plt.imshow(output_img)
    plt.title(f"File: {base}")
    plt.axis("off")
    plt.show()

    input("Press Enter for next image...")
