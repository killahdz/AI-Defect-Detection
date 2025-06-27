import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import yaml
from glob import glob
from ultralytics import YOLO

#model image size
model_image_size = 256

# Paths
pt_model_path = "C:/training/kaggle/models/wood_defect_img256_yolov8s.pt"
dataset_yaml_path = "C:/training/kaggle/dataset.yaml"
test_img_dir = "C:/training/kaggle/train/images"
test_label_dir = "C:/training/kaggle/train/labels"

# Load class names from dataset.yaml
with open(dataset_yaml_path, "r") as f:
    data = yaml.safe_load(f)
class_names = data["names"]
num_classes = len(class_names)
print(f"Loaded {num_classes} classes: {class_names}")

# Load YOLOv8 model
model = YOLO(pt_model_path)
print("Model loaded successfully")
print(f"Model classes: {model.names}")

# Load ground-truth annotations
def load_annotations(label_path, orig_width=2800, orig_height=1024, target_width=model_image_size, target_height=model_image_size):
    print(f"Checking label file: {label_path}")
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return []
    with open(label_path, "r") as f:
        lines = f.readlines()
    annotations = []
    for line in lines:
        if line.strip():
            try:
                cls, x_center, y_center, width, height = map(float, line.strip().split())
                print(f"Raw annotation: cls={cls}, x_center={x_center}, y_center={y_center}, width={width}, height={height}")
                # Scale to target size
                x = x_center * orig_width * (target_width / orig_width)
                y = y_center * orig_height * (target_height / orig_height)
                w = width * orig_width * (target_width / orig_width)
                h = height * orig_height * (target_height / orig_height)
                annotations.append((int(cls), x, y, w, h))
            except ValueError:
                print(f"Invalid annotation in {label_path}: {line.strip()}")
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

# Match predictions to ground-truth
def match_predictions(pred_boxes, pred_scores, pred_classes, gt_boxes, iou_thres=0.3):
    tp = np.zeros(num_classes, dtype=int)
    fp = np.zeros(num_classes, dtype=int)
    fn = np.zeros(num_classes, dtype=int)
    matched_gt = set()
    tp_matches = []  # Store (pred_idx, gt_idx) for TPs
    
    pred_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[pred_indices]
    pred_scores = pred_scores[pred_indices]
    pred_classes = pred_classes[pred_indices]
    
    print("Matching predictions to ground-truth:")
    for pred_idx, (box, score, cls) in enumerate(zip(pred_boxes, pred_scores, pred_classes)):
        x, y, w, h = box
        pred_box = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        print(f"Prediction {pred_idx}: Class={class_names[int(cls)]}, Score={score:.2f}, Box={pred_box}")
        best_iou = 0
        best_gt_idx = -1
        best_gt_cls = -1
        for gt_idx, (gt_cls, x, y, w, h) in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            gt_box = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
            iou = compute_iou(pred_box, gt_box)
            print(f"  GT {gt_idx}: Class={class_names[int(gt_cls)]}, Box={gt_box}, IoU={iou:.3f}")
            if iou > best_iou and cls == gt_cls:
                best_iou = iou
                best_gt_idx = gt_idx
                best_gt_cls = gt_cls
        print(f"  Best IoU={best_iou:.3f}, Best GT idx={best_gt_idx}, Best GT class={class_names[int(best_gt_cls)] if best_gt_cls != -1 else 'None'}")
        if best_iou >= iou_thres:
            tp[cls] += 1
            matched_gt.add(best_gt_idx)
            tp_matches.append((pred_idx, best_gt_idx))
        else:
            fp[cls] += 1
    
    for gt_idx, (gt_cls, _, _, _, _) in enumerate(gt_boxes):
        if gt_idx not in matched_gt:
            fn[gt_cls] += 1
            print(f"FN: GT {gt_idx}, Class={class_names[int(gt_cls)]}")
    
    print(f"TP={tp}, FP={fp}, FN={fn}")
    return tp, fp, fn, tp_matches

# Draw bounding boxes
def draw_boxes(img, pred_boxes, pred_scores, pred_classes, gt_boxes, tp_matches, display_width=1400, display_height=512):
    # Resize image to display size while preserving aspect ratio
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (display_width, display_height), interpolation=cv2.INTER_AREA)
    detection_color = (180, 0, 255)  # Magenta for non-matched predictions
    annotation_color = (0, 0, 255)   # Blue for non-matched GT
    match_color = (0, 255, 0)        # Green for TP matches
    
    # Scale factors from model_image_size x model_image_size to display size
    scale_x = display_width / model_image_size
    scale_y = display_height / model_image_size
    
    print(f"Number of ground-truth boxes: {len(gt_boxes)}")
    print(f"Ground-truth boxes ({model_image_size}x{model_image_size}): {gt_boxes}")
    
    # Scale predicted boxes
    scaled_pred_boxes = pred_boxes.copy()
    if pred_boxes.size > 0:
        scaled_pred_boxes[:, 0] *= scale_x  # x
        scaled_pred_boxes[:, 1] *= scale_y  # y
        scaled_pred_boxes[:, 2] *= scale_x  # w
        scaled_pred_boxes[:, 3] *= scale_y  # h
    
    # Scale GT boxes
    scaled_gt_boxes = [(cls, x * scale_x, y * scale_y, w * scale_x, h * scale_y) for cls, x, y, w, h in gt_boxes]
    
    print(f"Scaled ground-truth boxes ({display_width}x{display_height}): {scaled_gt_boxes}")
    
    # Draw predicted boxes
    matched_pred_indices = {pred_idx for pred_idx, _ in tp_matches}
    for pred_idx, (box, score, cls) in enumerate(zip(scaled_pred_boxes, pred_scores, pred_classes)):
        x, y, w, h = box
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        # Clip coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
        # Use green for TP, magenta for others
        color = match_color if pred_idx in matched_pred_indices else detection_color
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 1)
        label = f"{class_names[int(cls)]}: {score:.2f}"
        cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw GT boxes
    matched_gt_indices = {gt_idx for _, gt_idx in tp_matches}
    for gt_idx, (cls, x, y, w, h) in enumerate(scaled_gt_boxes):
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        # Clip coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
        # Use green for TP, blue for others
        color = match_color if gt_idx in matched_gt_indices else annotation_color
        print(f"Rendering GT box: cls={class_names[int(cls)]}, color={'Green' if color == match_color else 'Blue'}, x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 1)
        label = f"{class_names[int(cls)]}"
        cv2.putText(img_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_rgb

# Main inference loop
def main():
    image_paths = []
    for ext in [".jpg", ".png", ".jpeg"]:
        image_paths.extend(glob(os.path.join(test_img_dir, f"*{ext}")))
    if not image_paths:
        print("No images found in val or test directories")
        return
    print(f"Found {len(image_paths)} images for inference")
    print(f"Dataset classes: {class_names}")
    print(f"Model classes: {model.names}")

    tp_counts = np.zeros(num_classes, dtype=int)
    fp_counts = np.zeros(num_classes, dtype=int)
    fn_counts = np.zeros(num_classes, dtype=int)

    plt.ion()
    fig1, ax1 = plt.subplots(figsize=(14, 5.12))  # Aspect ratio ~2800/1024
    img_display = ax1.imshow(np.zeros((512, 1400, 3)))
    title = ax1.set_title("")
    ax1.axis('off')

    for idx, img_path in enumerate(image_paths, 1):
        try:
            print(f"\nProcessing image {idx}/{len(image_paths)}: {os.path.basename(img_path)}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            print(f"Image shape: {img.shape}")

            # Predict with imgsz=model_image_size
            results = model.predict(img_path, imgsz=model_image_size, conf=0.01, device='cpu')
            print(f"Original shape: {results[0].orig_shape}")
            pred_boxes = results[0].boxes.xywh.cpu().numpy()
            pred_scores = results[0].boxes.conf.cpu().numpy()
            pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)

            # Scale predicted boxes to model_image_size for IoU
            if pred_boxes.size > 0:
                pred_boxes[:, 0] *= model_image_size / results[0].orig_shape[1]  # x
                pred_boxes[:, 1] *= model_image_size / results[0].orig_shape[0]  # y
                pred_boxes[:, 2] *= model_image_size / results[0].orig_shape[1]  # w
                pred_boxes[:, 3] *= model_image_size / results[0].orig_shape[0]  # h

            label_path = img_path.replace("images", "labels").replace(os.path.splitext(img_path)[1], ".txt")
            gt_boxes = load_annotations(label_path, orig_width=2800, orig_height=1024, target_width=model_image_size, target_height=model_image_size)

            print("Ground Truth Boxes:", gt_boxes)
            print("Predicted Boxes:", [(class_names[int(cls)], box.tolist(), score) for box, score, cls in zip(pred_boxes, pred_scores, pred_classes)])

            tp, fp, fn, tp_matches = match_predictions(pred_boxes, pred_scores, pred_classes, gt_boxes)
            tp_counts += tp
            fp_counts += fp
            fn_counts += fn

            img_rgb = draw_boxes(img, pred_boxes, pred_scores, pred_classes, gt_boxes, tp_matches, display_width=1400, display_height=512)
            img_display.set_data(img_rgb)
            title.set_text(f"Image {idx}/{len(image_paths)}\n{os.path.basename(img_path)}")
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            plt.pause(1.5)

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    plt.ioff()
    plt.close('all')
    print("\nFinal Metrics:")
    for i in range(num_classes):
        precision = (tp_counts[i] / (tp_counts[i] + fp_counts[i]) * 100) if (tp_counts[i] + fp_counts[i]) > 0 else 0.0
        recall = (tp_counts[i] / (tp_counts[i] + fn_counts[i]) * 100) if (tp_counts[i] + fn_counts[i]) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        print(f"{class_names[i]}: TP={tp_counts[i]}, FP={fp_counts[i]}, FN={fn_counts[i]}, "
              f"Precision={precision:.1f}%, Recall={recall:.1f}%, F1-Score={f1:.1f}%")

if __name__ == "__main__":
    main()