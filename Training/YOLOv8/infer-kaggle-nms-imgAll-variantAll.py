import datetime
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import yaml
import random
from glob import glob
from ultralytics import YOLO
import seaborn as sns
from scipy.optimize import linear_sum_assignment

# Model image size
model_image_size = 512
model_variant = 's'

# Dataset choice: 'kaggle' or 'dataset-ninja'
dataset_choice = 'dataset-ninja'

# Dataset configurations
datasets = {
    'kaggle': {
        'pt_model_path': f"D:/training/{dataset_choice}/models/wood_defect_img{model_image_size}_yolov8{model_variant}.pt",
        'dataset_yaml_path': f"D:/training/{dataset_choice}/dataset.yaml",
        'test_img_dir': f"D:/training/{dataset_choice}/train/images",
        'test_label_dir': f"D:/training/{dataset_choice}/train/labels",
        'orig_width': 2800,
        'orig_height': 1024
    },
    'dataset-ninja': {
        'pt_model_path': f"D:/training/{dataset_choice}/models/wood_defect_img{model_image_size}_yolov8{model_variant}.pt",
        'dataset_yaml_path': f"D:/training/{dataset_choice}/dataset.yaml",
        'test_img_dir': f"D:/training/{dataset_choice}/train/images",
        'test_label_dir': f"D:/training/{dataset_choice}/train/labels",
        'orig_width': 2800,
        'orig_height': 1024
    }
}

# Select dataset configuration
config = datasets[dataset_choice]
pt_model_path = config['pt_model_path']
dataset_yaml_path = config['dataset_yaml_path']
test_img_dir = config['test_img_dir']
test_label_dir = config['test_label_dir']
orig_width = config['orig_width']
orig_height = config['orig_height']

# Load class names from Kaggle dataset.yaml
with open(dataset_yaml_path, "r") as f:
    data = yaml.safe_load(f)
class_names = data["names"]
num_classes = len(class_names)
print(f"Loaded {num_classes} classes from Kaggle dataset: {class_names}")

# Load YOLOv8 model
model = YOLO(pt_model_path)
print("Model loaded successfully")
print(f"Model classes: {model.names}")
if len(model.names) != num_classes or any(model.names[i] != class_names[i] for i in range(num_classes)):
    print(f"Warning: Model classes {model.names} do not match dataset classes {class_names}")

# Data-Ninja to Kaggle class name mapping
data_ninja_to_kaggle = {
    'Quartzity': 0,
    'Live_knot': 1,
    'Live_Knot': 1,
    'Marrow': 2,
    'Resin': 3,
    'resin': 3,
    'Dead_knot': 4,
    'Death_know': 4,
    'Dead_Knot': 4,
    'Knot_with_crack': 5,
    'knot_with_crack': 5,
    'Knot_missing': 6,
    'Crack': 7,
    'Blue_stain': -1,
    'Overgrown': -1
}

# Load ground-truth annotations
def load_annotations(img_path, test_label_dir, orig_width, orig_height, target_width=model_image_size, target_height=model_image_size):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(test_label_dir, img_name + ".txt")
    print(f"Checking label file: {label_path}")
    
    if not os.path.exists(label_path):
        base, _ = os.path.splitext(label_path)
        for ext in ['.txt', '.TXT', '.Txt']:
            alt_path = base + ext
            if os.path.exists(alt_path):
                label_path = alt_path
                print(f"Found alternative label file: {label_path}")
                break
        else:
            print(f"Label file not found: {label_path}")
            print(f"Available files in {test_label_dir}: {os.listdir(test_label_dir)[:10]}")
            return []
    
    annotations = []
    with open(label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.strip():
            try:
                parts = line.strip().split()
                if dataset_choice == 'kaggle':
                    cls = float(parts[0])
                    cls_name = class_names[int(cls)] if 0 <= int(cls) < num_classes else 'Unknown'
                else:
                    cls_name = parts[0]
                    cls = data_ninja_to_kaggle.get(cls_name, -1)
                    if cls == -1:
                        print(f"Ignoring annotation with class '{cls_name}' in {label_path}")
                        continue
                x_center, y_center, width, height = map(float, parts[1:5])
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    print(f"Out-of-bounds coordinates in {label_path}: {line.strip()}")
                    continue
                if height > 0.5 or width > 0.5:
                    print(f"Large box detected in {label_path}: cls={cls_name}, width={width}, height={height}")
                    width = min(width, 0.5)
                    height = min(height, 0.5)
                print(f"Raw annotation: cls={cls} ({cls_name}), x_center={x_center}, y_center={y_center}, width={width}, height={height}")
                x = x_center * target_width
                y = y_center * target_height
                w = width * target_width
                h = height * target_height
                annotations.append((int(cls), x, y, w, h))
            except (ValueError, IndexError) as e:
                print(f"Invalid annotation in {label_path}: {line.strip()} (Error: {str(e)})")
    print(f"Loaded {len(annotations)} annotations from {label_path}")
    return annotations

# Compute IoU
def compute_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

# Non-Maximum Suppression
def apply_nms(boxes, scores, classes, iou_thres=0.5):
    if len(boxes) == 0:
        return boxes, scores, classes
    
    keep_indices = []
    boxes_xyxy = np.array([[x - w/2, y - h/2, x + w/2, y + h/2] for x, y, w, h in boxes])
    
    for cls in range(num_classes):
        cls_mask = classes == cls
        if not np.any(cls_mask):
            continue
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]
        
        order = np.argsort(cls_scores)[::-1]
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]
        cls_indices = cls_indices[order]
        
        while len(cls_boxes) > 0:
            keep_indices.append(cls_indices[0])
            if len(cls_boxes) == 1:
                break
            ious = np.array([compute_iou(cls_boxes[0], box) for box in cls_boxes[1:]])
            keep_mask = ious < iou_thres
            cls_boxes = cls_boxes[1:][keep_mask]
            cls_scores = cls_scores[1:][keep_mask]
            cls_indices = cls_indices[1:][keep_mask]
    
    keep_indices = np.array(keep_indices)
    return boxes[keep_indices], scores[keep_indices], classes[keep_indices]

# Match predictions to ground-truth
def match_predictions_hungarian(pred_boxes, pred_scores, pred_classes, gt_boxes, iou_thres=0.5):
    tp = np.zeros(num_classes, dtype=int)
    fp = np.zeros(num_classes, dtype=int)
    fn = np.zeros(num_classes, dtype=int)
    tp_matches = []

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        if len(gt_boxes) > 0:
            for gt_cls, _, _, _, _ in gt_boxes:
                fn[gt_cls] += 1
        if len(pred_boxes) > 0:
            for cls in pred_classes:
                fp[cls] += 1
        return tp, fp, fn, tp_matches, []

    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, (pbox, pcls) in enumerate(zip(pred_boxes, pred_classes)):
        px, py, pw, ph = pbox
        pred_box = [px - pw/2, py - ph/2, px + pw/2, py + ph/2]
        for j, (gcls, gx, gy, gw, gh) in enumerate(gt_boxes):
            gt_box = [gx - gw/2, gy - gh/2, gx + gw/2, gy + gh/2]
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_thres and pcls == gcls:
                iou_matrix[i, j] = iou
            else:
                iou_matrix[i, j] = 0

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Maximize IoU (minimize -IoU)
    matched_gt = set()
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] > 0:
            cls = pred_classes[i]
            tp[cls] += 1
            matched_gt.add(j)
            tp_matches.append((i, j))

    # Count FPs
    matched_pred = set(row_ind[iou_matrix[row_ind, col_ind] > 0])
    for i, cls in enumerate(pred_classes):
        if i not in matched_pred:
            fp[cls] += 1

    # Count FNs
    for j, (gt_cls, _, _, _, _) in enumerate(gt_boxes):
        if j not in matched_gt:
            fn[gt_cls] += 1

    return tp, fp, fn, tp_matches, []

# Draw bounding boxes
def draw_boxes(img, pred_boxes, pred_scores, pred_classes, gt_boxes, tp_matches, partial_matches, display_width=1400, display_height=512):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (display_width, display_height), interpolation=cv2.INTER_AREA)
    detection_color = (180, 0, 255)  # Pink for unmatched predictions
    annotation_color = (0, 0, 255)    # Blue for unmatched ground-truth
    match_color = (0, 255, 0)         # Green for true positives
    partial_color = (225, 255, 0)     # Orange for partial matches
    
    scale_x = display_width / model_image_size
    scale_y = display_height / model_image_size
    
    print(f"Number of ground-truth boxes: {len(gt_boxes)}")
    print(f"Ground-truth boxes ({model_image_size}x{model_image_size}): {gt_boxes}")
    
    scaled_pred_boxes = pred_boxes.copy()
    if pred_boxes.size > 0:
        scaled_pred_boxes[:, 0] *= scale_x
        scaled_pred_boxes[:, 1] *= scale_y
        scaled_pred_boxes[:, 2] *= scale_x
        scaled_pred_boxes[:, 3] *= scale_y
    
    scaled_gt_boxes = [(cls, x * scale_x, y * scale_y, w * scale_x, h * scale_y) for cls, x, y, w, h in gt_boxes]
    print(f"Scaled ground-truth boxes ({display_width}x{display_height}): {scaled_gt_boxes}")
    
    matched_pred_indices = {pred_idx for pred_idx, _ in tp_matches}
    partial_pred_indices = {pred_idx for pred_idx, _ in partial_matches}
    for pred_idx, (box, score, cls) in enumerate(zip(scaled_pred_boxes, pred_scores, pred_classes)):
        x, y, w, h = box
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_rgb.shape[1] - 1, x2), min(img_rgb.shape[0] - 1, y2)
        if pred_idx in matched_pred_indices:
            color = match_color
        elif pred_idx in partial_pred_indices:
            color = partial_color
        else:
            color = detection_color
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 1)
        label = f"{class_names[int(cls)]}: {score:.2f}"
        cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    matched_gt_indices = {gt_idx for _, gt_idx in tp_matches}
    partial_gt_indices = {gt_idx for _, gt_idx in partial_matches}
    for gt_idx, (cls, x, y, w, h) in enumerate(scaled_gt_boxes):
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_rgb.shape[1] - 1, x2), min(img_rgb.shape[0] - 1, y2)
        if gt_idx in matched_gt_indices:
            color = match_color
        elif gt_idx in partial_gt_indices:
            color = partial_color
        else:
            color = annotation_color
        print(f"Rendering GT box: cls={class_names[int(cls)]}, color={'Green' if color == match_color else 'Orange' if color == partial_color else 'Blue'}, x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 1)
        label = f"{class_names[int(cls)]}"
        cv2.putText(img_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_rgb

# Display performance metrics as a heatmap in real-time
def update_metrics_heatmap(ax, heatmap, class_names, precision, recall, f1, image_idx, total_images):
    metrics = np.array([precision, recall, f1]).T  # Shape: (num_classes, 3)
    ax.clear()  # Clear previous heatmap
    sns.heatmap(
        metrics,
        annot=True,
        fmt='.1f',
        cmap='YlGnBu',
        xticklabels=['Precision (%)', 'Recall (%)', 'F1-Score (%)'],
        yticklabels=class_names,
        cbar=False,
        ax=ax
    )
    ax.set_title(f'Performance Metrics per Class (Image {image_idx}/{total_images})')
    plt.draw()
    plt.pause(0.1)  # Brief pause to allow rendering

# Main inference loop
def main():
    image_paths = []
    for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        image_paths.extend(glob(os.path.join(test_img_dir, f"*{ext}")))
    if not image_paths:
        print(f"No images found in {test_img_dir}")
        return
    print(f"Found {len(image_paths)} images for inference in {dataset_choice} dataset")
    print(f"Dataset classes: {class_names}")
    print(f"Model classes: {model.names}")

    tp_counts = np.zeros(num_classes, dtype=int)
    fp_counts = np.zeros(num_classes, dtype=int)
    fn_counts = np.zeros(num_classes, dtype=int)
    precision_list = np.zeros(num_classes)
    recall_list = np.zeros(num_classes)
    f1_list = np.zeros(num_classes)

    plt.ion()
    # Image visualization figure
    fig1, ax1 = plt.subplots(figsize=(14, 5.12))
    img_display = ax1.imshow(np.zeros((512, 1400, 3)))
    title = ax1.set_title("")
    ax1.axis('off')

    # Heatmap figure
    fig2, ax2 = plt.subplots(figsize=(8, len(class_names) * 0.5))
    heatmap = None  # Placeholder for initial heatmap

    enumerated_image_paths = list(enumerate(image_paths, 1))
    random.shuffle(enumerated_image_paths)    
    for idx, img_path in enumerated_image_paths:
        try:
            print(f"\nProcessing image {idx}/{len(image_paths)}: {os.path.basename(img_path)}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            print(f"Image shape: {img.shape}")

            results = model.predict(img_path, imgsz=model_image_size, conf=0.25, device='cpu')
            print(f"Original shape: {results[0].orig_shape}")
            pred_boxes = results[0].boxes.xywh.cpu().numpy()
            pred_scores = results[0].boxes.conf.cpu().numpy()
            pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)

            if pred_boxes.size > 0:
                pred_boxes[:, 0] *= model_image_size / results[0].orig_shape[1]
                pred_boxes[:, 1] *= model_image_size / results[0].orig_shape[0]
                pred_boxes[:, 2] *= model_image_size / results[0].orig_shape[1]
                pred_boxes[:, 3] *= model_image_size / results[0].orig_shape[0]

                # Apply NMS to filter overlapping predictions
                pred_boxes, pred_scores, pred_classes = apply_nms(pred_boxes, pred_scores, pred_classes, iou_thres=0.5)
                print(f"After NMS: {len(pred_boxes)} predictions remain")

            gt_boxes = load_annotations(img_path, test_label_dir, orig_width=orig_width, orig_height=orig_height, target_width=model_image_size, target_height=model_image_size)

            print("Ground Truth Boxes:", gt_boxes)
            print("Predicted Boxes:", [(class_names[int(cls)], box.tolist(), score) for box, score, cls in zip(pred_boxes, pred_scores, pred_classes)])

            tp, fp, fn, tp_matches, partial_matches = match_predictions_hungarian(pred_boxes, pred_scores, pred_classes, gt_boxes)
            tp_counts += tp
            fp_counts += fp
            fn_counts += fn

            # Update metrics
            for i in range(num_classes):
                precision_list[i] = (tp_counts[i] / (tp_counts[i] + fp_counts[i]) * 100) if (tp_counts[i] + fp_counts[i]) > 0 else 0.0
                recall_list[i] = (tp_counts[i] / (tp_counts[i] + fn_counts[i]) * 100) if (tp_counts[i] + fn_counts[i]) > 0 else 0.0
                f1_list[i] = (2 * precision_list[i] * recall_list[i] / (precision_list[i] + recall_list[i])) if (precision_list[i] + recall_list[i]) > 0 else 0.0

            # Update image visualization
            img_rgb = draw_boxes(img, pred_boxes, pred_scores, pred_classes, gt_boxes, tp_matches, partial_matches, display_width=1400, display_height=512)
            img_display.set_data(img_rgb)
            title.set_text(f"Image {idx}/{len(image_paths)}\n{os.path.basename(img_path)}")
            fig1.canvas.draw()
            fig1.canvas.flush_events()

            # Update metrics heatmap
            update_metrics_heatmap(ax2, heatmap, class_names, precision_list, recall_list, f1_list, idx, len(image_paths))
            fig2.canvas.draw()
            fig2.canvas.flush_events()
            plt.pause(0.5)

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    plt.ioff()
    plt.close('all')
    print("\nFinal Metrics:")
    for i in range(num_classes):
        print(f"{class_names[i]}: TP={tp_counts[i]}, FP={fp_counts[i]}, FN={fn_counts[i]}, "
              f"Precision={precision_list[i]:.1f}%, Recall={recall_list[i]:.1f}%, F1-Score={f1_list[i]:.1f}%")


if __name__ == "__main__":
    main()