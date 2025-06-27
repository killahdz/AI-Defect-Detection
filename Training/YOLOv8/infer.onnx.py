import onnxruntime as ort
import cv2
import numpy as np
import os
import yaml
import time
from glob import glob

# Paths
onnx_model_path = "C:/training/kaggle/models/best.onnx"
dataset_yaml_path = "C:/training/kaggle/dataset.yaml"
val_img_dir = "C:/training/kaggle/val/images"
val_label_dir = "C:/training/kaggle/val/labels"
test_img_dir = "C:/training/kaggle/train/images"
test_label_dir = "C:/training/kaggle/train/labels"


# Verify ONNX model exists
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

# Load class names from dataset.yaml
if not os.path.exists(dataset_yaml_path):
    raise FileNotFoundError(f"dataset.yaml not found at {dataset_yaml_path}")
with open(dataset_yaml_path, "r") as f:
    data = yaml.safe_load(f)
class_names = data["names"]
num_classes = len(class_names)
print(f"Loaded {num_classes} classes: {class_names}")

# Initialize ONNX session
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
print("ONNX model loaded successfully")

# Image preprocessing function
def preprocess_image(image_path, img_size=128):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    orig_img = img.copy()
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img, orig_img

# Post-processing function for YOLOv8 ONNX output
def postprocess(output, img_size=128, conf_thres=0.1, iou_thres=0.45):  # Lowered conf_thres
    try:
        predictions = output[0]  # Shape: [1, 84, 8400] for YOLOv8n
        print(f"Output shape: {predictions.shape}")  # Debug output shape
        predictions = predictions.transpose(0, 2, 1)  # [1, 8400, 84]
        boxes = predictions[..., :4]  # x1, y1, x2, y2
        scores = predictions[..., 4:5]  # Confidence scores
        class_probs = predictions[..., 5:]  # Class probabilities

        # Filter by confidence
        conf = scores * class_probs.max(axis=-1, keepdims=True)
        mask = conf[..., 0] > conf_thres
        if not np.any(mask):
            print("No detections above confidence threshold")
            return np.array([]), np.array([]), np.array([])

        boxes = boxes[mask]
        conf = conf[mask]
        cls = class_probs[mask].argmax(axis=-1)

        # Convert boxes from [x1, y1, x2, y2] to [x, y, w, h]
        boxes = np.concatenate(
            [(boxes[..., 0:2] + boxes[..., 2:4]) / 2, boxes[..., 2:4] - boxes[..., 0:2]], axis=-1
        )
        # Scale boxes to image size
        boxes *= img_size
        scores = conf.max(axis=-1)

        # Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), conf_thres, iou_thres
        )
        if isinstance(indices, np.ndarray) and len(indices) > 0:
            indices = indices.flatten()
        else:
            print("No detections after NMS")
            return np.array([]), np.array([]), np.array([])

        return boxes[indices], scores[indices], cls[indices]
    except Exception as e:
        print(f"Post-processing error: {str(e)}")
        return np.array([]), np.array([]), np.array([])

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

# Draw bounding boxes and labels
def draw_boxes(img, boxes, scores, classes, annotations, class_names):
    # Draw predicted boxes (green)
    for box, score, cls in zip(boxes, scores, classes):
        x, y, w, h = box
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[int(cls)]}: {score:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw ground-truth boxes (red)
    for cls, x1, y1, x2, y2 in annotations:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        label = f"{class_names[int(cls)]} (GT)"
        cv2.putText(img, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img

# Main inference loop
def main():
    # Verify directories exist
    for dir_path in [val_img_dir, test_img_dir]:
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            return

    # Collect image paths
    image_paths = []
    for ext in [".jpg", ".png", ".jpeg"]:  # Add ".bmp" if needed
        image_paths.extend(glob(os.path.join(val_img_dir, f"*{ext}")))
        image_paths.extend(glob(os.path.join(test_img_dir, f"*{ext}")))

    if not image_paths:
        print("No images found in val or test directories")
        return

    print(f"Found {len(image_paths)} images for inference")

    # Process each image
    for idx, img_path in enumerate(image_paths, 1):
        try:
            print(f"\nProcessing image {idx}/{len(image_paths)}: {os.path.basename(img_path)}")

            # Preprocess image
            input_img, orig_img = preprocess_image(img_path)

            # Run inference
            inputs = {session.get_inputs()[0].name: input_img}
            outputs = session.run(None, inputs)

            # Post-process predictions
            boxes, scores, classes = postprocess(outputs, img_size=128)

            # Load ground-truth annotations
            label_path = img_path.replace("images", "labels").replace(os.path.splitext(img_path)[1], ".txt")
            annotations = load_annotations(label_path, img_size=128)

            # Print actual vs predicted
            print("Ground Truth:")
            if annotations:
                for cls, x1, y1, x2, y2 in annotations:
                    print(f"  Class: {class_names[int(cls)]}, Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            else:
                print("  No ground-truth annotations")
            print("Predictions:")
            if len(boxes) > 0:
                for box, score, cls in zip(boxes, scores, classes):
                    x, y, w, h = box
                    print(f"  Class: {class_names[int(cls)]}, Score: {score:.2f}, Box: [{x-w/2:.1f}, {y-h/2:.1f}, {x+w/2:.1f}, {y+h/2:.1f}]")
            else:
                print("  No predictions")

            # Draw and display image
            display_img = draw_boxes(orig_img.copy(), boxes, scores, classes, annotations, class_names)
            cv2.imshow("YOLOv8 Inference", display_img)
            cv2.waitKey(1000)  # Display for 1 second
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    cv2.destroyAllWindows()
    print("\nInference completed")

if __name__ == "__main__":
    main()