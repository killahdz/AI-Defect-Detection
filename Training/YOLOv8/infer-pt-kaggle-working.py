import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import random
import yaml
from glob import glob
from ultralytics import YOLO


# Paths
pt_model_path = "C:/training/kaggle/models/wood_defect_yolov8s.pt"
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
def load_annotations(label_path, img_size=128):
    if not os.path.exists(label_path):
        return []
    with open(label_path, "r") as f:
        lines = f.readlines()
    annotations = []
    for line in lines:
        if line.strip():
            try:
                cls, x_center, y_center, width, height = map(float, line.strip().split())
                x1 = (x_center - width / 2) * img_size
                y1 = (y_center - height / 2) * img_size
                x2 = (x_center + width / 2) * img_size
                y2 = (y_center + height / 2) * img_size
                annotations.append((int(cls), x1, y1, x2, y2))
            except ValueError:
                print(f"Invalid annotation in {label_path}: {line.strip()}")
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

# Match predictions to ground-truth
def match_predictions(pred_boxes, pred_scores, pred_classes, gt_boxes, iou_thres=0.5):
    correct = np.zeros(num_classes, dtype=int)
    incorrect = np.zeros(num_classes, dtype=int)
    matched_gt = set()
    for pred_idx, (box, score, cls) in enumerate(zip(pred_boxes, pred_scores, pred_classes)):
        x, y, w, h = box
        pred_box = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, (gt_cls, x1, y1, x2, y2) in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            gt_box = [x1, y1, x2, y2]
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou and cls == gt_cls:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= iou_thres:
            correct[cls] += 1
            matched_gt.add(best_gt_idx)
        else:
            incorrect[cls] += 1
    for gt_idx, (gt_cls, _, _, _, _) in enumerate(gt_boxes):
        if gt_idx not in matched_gt:
            incorrect[gt_cls] += 1
    return correct, incorrect

# Draw bounding boxes
def draw_boxes(img, pred_boxes, pred_scores, pred_classes, gt_boxes):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detection_color = (180, 0, 255) 
    annotation_color =  (0, 0, 255)
    # Draw detections 
    for box, score, cls in zip(pred_boxes, pred_scores, pred_classes):
        x, y, w, h = box
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), detection_color, 2)
        label = f"{class_names[int(cls)]}: {score:.2f}"
        cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, detection_color, 2)

    # Draw ground truth boxes 
    for cls, x1, y1, x2, y2 in gt_boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        print(f"Rendering GT box: cls={class_names[int(cls)]}, x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), annotation_color, 2)  # Thickness 1
        label = f"[{class_names[int(cls)]}]"
        cv2.putText(img_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, annotation_color, 2)  # Font scale 0.3, thickness 1
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
    random.shuffle(image_paths)

    # Initialize counters
    correct_counts = np.zeros(num_classes, dtype=int)
    incorrect_counts = np.zeros(num_classes, dtype=int)

    # Set up interactive Matplotlib figures
    plt.ion()
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    img_display = ax1.imshow(np.zeros((128, 128, 3)))  # Placeholder
    title = ax1.set_title("")
    ax1.axis('off')

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.axis('off')
    table_data = [[class_names[i], 0, 0, 0.0] for i in range(num_classes)] + [['Total', 0, 0, 0.0]]
    table = ax2.table(cellText=table_data,
                      colLabels=['Class', 'Correct', 'Incorrect', 'Success Rate (%)'],
                      loc='center', cellLoc='center', colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Process each image
    total_correct = 0
    total_incorrect = 0
    
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

            # Run inference
            results = model.predict(img_path, imgsz=128, conf=0.01, device='cpu')
            pred_boxes = results[0].boxes.xywh.cpu().numpy()
            pred_scores = results[0].boxes.conf.cpu().numpy()
            pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)

            # Load ground-truth annotations
            label_path = img_path.replace("images", "labels").replace(os.path.splitext(img_path)[1], ".txt")
            gt_boxes = load_annotations(label_path, img_size=128)

            # Match predictions to ground-truth
            correct, incorrect = match_predictions(pred_boxes, pred_scores, pred_classes, gt_boxes)
            correct_counts += correct
            incorrect_counts += incorrect
            total_correct += correct.sum()
            total_incorrect += incorrect.sum()

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
                table.get_celld()[(j+1, 1)].get_text().set_text(str(correct_counts[j]))
                table.get_celld()[(j+1, 2)].get_text().set_text(str(incorrect_counts[j]))
                total_j = correct_counts[j] + incorrect_counts[j]
                success_rate = (correct_counts[j] / total_j * 100) if total_j > 0 else 0.0
                table.get_celld()[(j+1, 3)].get_text().set_text(f"{success_rate:.1f}")
            table.get_celld()[(num_classes+1, 1)].get_text().set_text(str(total_correct))
            table.get_celld()[(num_classes+1, 2)].get_text().set_text(str(total_incorrect))
            total_all = total_correct + total_incorrect
            total_success_rate = (total_correct / total_all * 100) if total_all > 0 else 0.0
            table.get_celld()[(num_classes+1, 3)].get_text().set_text(f"{total_success_rate:.1f}")
            fig2.canvas.draw()
            fig2.canvas.flush_events()

            plt.pause(0.5)  # Display for 0.5 seconds

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    plt.ioff()
    plt.close('all')
    print("\nFinal Tally:")
    for i in range(num_classes):
        total_i = correct_counts[i] + incorrect_counts[i]
        success_rate = (correct_counts[i] / total_i * 100) if total_i > 0 else 0.0
        print(f"{class_names[i]}: Correct={correct_counts[i]}, Incorrect={incorrect_counts[i]}, Success Rate={success_rate:.1f}%")
    print(f"Total: Correct={total_correct}, Incorrect={total_incorrect}, Success Rate={total_success_rate:.1f}%")
    print("Inference completed" if idx == len(image_paths) else f"Stopped early at image {idx}.")

if __name__ == "__main__":
    main()