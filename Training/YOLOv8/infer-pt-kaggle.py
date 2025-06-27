import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import yaml
from glob import glob
from ultralytics import YOLO

# Paths
pt_model_path = "C:/training/kaggle/models/wood_defect_yolov8n.pt"
dataset_yaml_path = "C:/training/kaggle/dataset.yaml"
val_img_dir = "C:/training/kaggle/val/images"
val_label_dir = "C:/training/kaggle/val/labels"
test_img_dir = "C:/training/kaggle/train/images"
test_label_dir = "C:/training/kaggle/train/labels"

# Verify model exists
if not os.path.exists(pt_model_path):
    raise FileNotFoundError(f"PyTorch model not found at {pt_model_path}")

# Load class names from dataset.yaml
if not os.path.exists(dataset_yaml_path):
    raise FileNotFoundError(f"dataset.yaml not found at {dataset_yaml_path}")
with open(dataset_yaml_path, "r") as f:
    data = yaml.safe_load(f)
class_names = data["names"]
num_classes = len(class_names)
print(f"Loaded {num_classes} classes: {class_names}")

# Load YOLOv8 model
model = YOLO(pt_model_path)
print("PyTorch model loaded successfully")

# Load ground-truth annotations
def load_annotations(label_path, img, expected_img_size=None):
    if not os.path.exists(label_path):
        print(f"Annotation file not found: {label_path}")
        return []
    # Get original image dimensions
    orig_height, orig_width = img.shape[:2]
    # Use expected image size from dataset if provided, otherwise use actual image size
    norm_width, norm_height = expected_img_size if expected_img_size else (orig_width, orig_height)
    print(f"Normalizing annotations using size: {norm_width}x{norm_height}")
    annotations = []
    with open(label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.strip():
            try:
                cls, x_center, y_center, width, height = map(float, line.strip().split())
                # Debug: Print raw annotation values
                print(f"Raw annotation: cls={cls}, x_center={x_center}, y_center={y_center}, width={width}, height={height}")
                # Check if annotations are normalized
                is_normalized = (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                0 <= width <= 1 and 0 <= height <= 1)
                if not is_normalized:
                    print(f"Non-normalized coordinates detected in {label_path}. Normalizing using {norm_width}x{norm_height}")
                    x_center /= norm_width
                    y_center /= norm_height
                    width /= norm_width
                    height /= norm_height
                # Validate normalized coordinates
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and width > 0 and height > 0):
                    print(f"Invalid coordinates after normalization in {label_path}: {line.strip()}")
                    continue
                # Scale to inference image size (128x128)
                x1 = (x_center - width / 2) * 128
                y1 = (y_center - height / 2) * 128
                x2 = (x_center + width / 2) * 128
                y2 = (y_center + height / 2) * 128
                # Ensure minimum box size for visibility
                min_box_size = 5.0  # 5 pixels for small boxes
                if x2 - x1 < min_box_size:
                    x1 -= (min_box_size - (x2 - x1)) / 2
                    x2 += (min_box_size - (x2 - x1)) / 2
                if y2 - y1 < min_box_size:
                    y1 -= (min_box_size - (y2 - y1)) / 2
                    y2 += (min_box_size - (y2 - y1)) / 2
                # Debug: Print pre-clamped coordinates
                print(f"Pre-clamped coordinates: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
                # Clamp coordinates to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(128, x2), min(128, y2)
                # Debug: Print final coordinates
                print(f"Final coordinates: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
                # Skip boxes that are too small after clamping
                if x2 - x1 < 1 or y2 - y1 < 1:
                    print(f"Skipping box too small after clamping: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                    continue
                annotations.append((int(cls), x1, y1, x2, y2))
            except ValueError:
                print(f"Invalid annotation format in {label_path}: {line.strip()}")
    return annotations

# Compute IoU between two boxes
def compute_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_2, y2_1)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

# Draw bounding boxes
def draw_boxes(img, pred_boxes, pred_scores, pred_classes, gt_boxes):
    # Resize image to match inference size (128x128)
    img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Draw predicted boxes (green)
    print(f"Number of predicted boxes: {len(pred_boxes)}")

    # Scale to match resized (128x128) image
    scale_x = 128 / img.shape[1]
    scale_y = 128 / img.shape[0]

    for box, score, cls in zip(pred_boxes, pred_scores, pred_classes):
        x1, y1, x2, y2 = map(float, box)
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y
        print(f"Rendering predicted box: cls={class_names[int(cls)]}, x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}, score={score:.2f}")
        cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        label = f"{class_names[int(cls)]}: {score:.2f}"
        cv2.putText(img_rgb, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)


    # Draw ground truth boxes (red)
    for cls, x1, y1, x2, y2 in gt_boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(f"Rendering GT box: cls={class_names[int(cls)]}, x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Thickness 1
        label = f"{class_names[int(cls)]} (GT)"
        cv2.putText(img_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)  # Font scale 0.3, thickness 1
    return img_rgb

# Main inference loop
def main():
    # Verify directories exist
    for dir_path in [val_img_dir, test_img_dir]:
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            return

    # Collect image paths
    image_paths = []
    for ext in [".jpg", ".png", ".jpeg"]:
        image_paths.extend(glob(os.path.join(val_img_dir, f"*{ext}")))
        image_paths.extend(glob(os.path.join(test_img_dir, f"*{ext}")))
    if not image_paths:
        print("No images found in val or test directories")
        return
    print(f"Found {len(image_paths)} images for inference")

    # Try to get expected image size from dataset.yaml (optional)
    expected_img_size = None
    try:
        with open(dataset_yaml_path, "r") as f:
            data = yaml.safe_load(f)
            if "img_size" in data:
                expected_img_size = data["img_size"]
                print(f"Using expected image size from dataset.yaml: {expected_img_size}")
    except Exception as e:
        print(f"Could not read image size from dataset.yaml: {e}")

    # Initialize counters
    gt_counts = np.zeros(num_classes, dtype=int)
    det_counts = np.zeros(num_classes, dtype=int)

    # Set up interactive Matplotlib figures
    plt.ion()
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    img_display = ax1.imshow(np.zeros((128, 128, 3)))  # Placeholder
    title = ax1.set_title("")
    ax1.axis('off')

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.axis('off')
    table_data = [[class_names[i], 0, 0] for i in range(num_classes)] + [['Total', 0, 0]]
    table = ax2.table(cellText=table_data,
                      colLabels=['Class', 'Ground Truth', 'Detected'],
                      loc='center', cellLoc='center', colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Process each image
    total_gt = 0
    total_det = 0
    for idx, img_path in enumerate(image_paths, 1):
        if not plt.get_fignums():
            print(f"Stopped at image {idx} due to window close.")
            break
        try:
            print(f"\nProcessing image {idx}/{len(image_paths)}: {os.path.basename(img_path)}")

            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            print(f"Original image size: {img.shape[:2]}")

            # Run inference
            results = model.predict(img_path, imgsz=128, conf=0.01, device='cpu')
            pred_boxes = results[0].boxes.xywh.cpu().numpy()
            pred_scores = results[0].boxes.conf.cpu().numpy()
            pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)

            # Load ground-truth annotations
            label_path = img_path.replace("images", "labels").replace(os.path.splitext(img_path)[1], ".txt")
            print(f"Loading annotations from: {label_path}")
            gt_boxes = load_annotations(label_path, img, expected_img_size)

            # Update ground truth counts
            for cls, _, _, _, _ in gt_boxes:
                gt_counts[int(cls)] += 1
                total_gt += 1

            # Update detected counts
            for cls in pred_classes:
                det_counts[int(cls)] += 1
                total_det += 1

            # Print ground-truth and predictions
            print("Ground Truth:")
            if gt_boxes:
                for cls, x1, y1, x2, y2 in gt_boxes:
                    print(f"  Class: {class_names[int(cls)]}, Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            else:
                print("  No ground-truth annotations")
            print("Predictions:")
            if len(pred_boxes) > 0:
                for box, score, cls in zip(pred_boxes, pred_scores, pred_classes):
                    x, y, w, h = box
                    print(f"  Class: {class_names[int(cls)]}, Score: {score:.2f}, Box: [{x-w/2:.1f}, {y-h/2:.1f}, {x+w/2:.1f}, {y+h/2:.1f}]")
            else:
                print("  No predictions")

            # Update image viewer
            img_rgb = draw_boxes(img, pred_boxes, pred_scores, pred_classes, gt_boxes)
            img_display.set_data(img_rgb)
            title.set_text(f"Image {idx}/{len(image_paths)}\n{os.path.basename(img_path)}")
            fig1.canvas.draw()
            fig1.canvas.flush_events()

            # Update tally grid
            for j in range(num_classes):
                table.get_celld()[(j+1, 1)].get_text().set_text(str(gt_counts[j]))
                table.get_celld()[(j+1, 2)].get_text().set_text(str(det_counts[j]))
            table.get_celld()[(num_classes+1, 1)].get_text().set_text(str(total_gt))
            table.get_celld()[(num_classes+1, 2)].get_text().set_text(str(total_det))
            fig2.canvas.draw()
            fig2.canvas.flush_events()

            plt.pause(0.5)  # Display for 0.5 seconds

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    plt.ioff()
    plt.close('all')
    print("\nFinal Tally:")
    for i in range(num_classes):
        print(f"{class_names[i]}: Ground Truth={gt_counts[i]}, Detected={det_counts[i]}")
    print(f"Total: Ground Truth={total_gt}, Detected={total_det}")
    print("Inference completed" if idx == len(image_paths) else f"Stopped early at image {idx}.")

if __name__ == "__main__":
    main()