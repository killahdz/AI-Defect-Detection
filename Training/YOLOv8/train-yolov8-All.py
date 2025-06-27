from ultralytics import YOLO
import os
import glob
import torch
import traceback
import yaml

# Summary of YOLOv8 model variants for wood defect detection with 2800x1200 images:
# - yolov8n.pt (Nano): Smallest (~3.2M params), fastest, for edge devices (e.g., Jetson Nano). Best with 960x960, but lower accuracy for small defects.
# - yolov8s.pt (Small): ~11.2M params, balances speed and accuracy. Suitable for 960x960 or 1280x1280 on mid-range GPUs (e.g., RTX 3060).
# - yolov8m.pt (Medium): ~25.9M params, your current choice, strong accuracy for small defects. Ideal for 1280x1280 on mid-to-high-end GPUs.
# - yolov8l.pt (Large): ~43.7M params, high accuracy for complex defects. Use with 1280x1280 on powerful GPUs (e.g., A100).
# - yolov8x.pt (Extra Large): ~68.2M params, maximum accuracy, but slow and resource-heavy. Best for 1280x1280 with top-tier GPUs, not ideal for real-time.
# Recommendation: Stick with yolov8m.pt for 1280x1280 for best accuracy-speed balance; use yolov8s.pt for 960x960 on limited hardware; optimize with TensorRT/INT8 for production.

# Configuration
base_path = "C:/training/dataset-ninja"
model_variant = "yolov8s"
img_size = 512
batch_size = 32
epochs = 50

# Verify dataset.yaml and load test path
yaml_path = f"{base_path}/dataset.yaml"
if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"dataset.yaml not found at {base_path}")

try:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_yaml = yaml.safe_load(f)
    test_dir = os.path.join(base_path, data_yaml['test']) if 'test' in data_yaml else None
except Exception as e:
    raise ValueError(f"Failed to parse dataset.yaml: {str(e)}")

# Check GPU
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected, using CPU")

# Load pretrained model
model = YOLO(f"{model_variant}.pt")

# Train
try:
    model.train(
        data=yaml_path,              # Path to dataset YAML file defining train/val/test paths and class names (e.g., D:/training/dataset-ninja/dataset.yaml)
        epochs=epochs,               # Number of training epochs (50 passes through the entire training dataset)
        imgsz=img_size,              # Input image size (512x512 pixels); affects model input resolution and detection scale
        batch=batch_size,            # Batch size (32 images per batch); balances memory usage and training stability
        name="wood_defect_model",    # Name for the training run; used for saving checkpoints and logs (e.g., wood_defect_model in wood_defect_runs)
        save=True,                   # Save trained model checkpoints (True enables saving best and last weights)
        save_period=10,              # Save checkpoint every 10 epochs; useful for resuming or analyzing intermediate models
        augment=True,                # Enable data augmentation (e.g., flips, rotations) to improve model generalization
        amp=True,                    # Enable Automatic Mixed Precision; reduces memory usage and speeds up training on GPUs
        lr0=0.001,                   # Initial learning rate (0.001); controls step size for weight updates
        cos_lr=True,                 # Use cosine learning rate schedule; gradually reduces learning rate for better convergence
        patience=20,                 # Early stopping patience (stop if no improvement after 20 epochs); prevents overfitting
        hsv_h=0.015,                 # Hue augmentation range (±1.5% hue shift); enhances color robustness for wood defects
        hsv_s=0.7,                   # Saturation augmentation range (±70% saturation shift); improves robustness to lighting variations
        hsv_v=0.4,                   # Value (brightness) augmentation range (±40% brightness shift); handles varying sawmill lighting
        degrees=10,                  # Rotation augmentation range (±10 degrees); improves robustness to rotated defects
        mosaic=1.0,                  # Probability of mosaic augmentation (1.0 = always apply); combines 4 images to improve context learning
        mixup=0.1,                   # Probability of mixup augmentation (10% chance); blends images/labels to enhance generalization
        project="wood_defect_runs",  # Directory to save training runs (e.g., wood_defect_runs/trainX for logs, weights, plots)
        plots=True,                  # Generate training plots (e.g., loss, mAP@0.5); saved in project directory for analysis
        cache='disk'                  # Cache images in RAM for faster data loading; 'ram' speeds up training if memory is sufficient
    )
    
except Exception as e:
    print(f"Training failed: {str(e)}\n{traceback.format_exc()}")

# Save PyTorch model
pt_save_path = f"{base_path}/models/wood_defect_img{img_size}_{model_variant}.pt"
os.makedirs(os.path.dirname(pt_save_path), exist_ok=True)
model.save(pt_save_path)
print(f"PyTorch model saved to {pt_save_path}")

# Export to ONNX
onnx_save_path = f"{base_path}/models/wood_defect_img{img_size}_{model_variant}.onnx"
try:
    model.export(format="onnx", imgsz=img_size, dynamic=False)
    print(f"ONNX model saved to {onnx_save_path}")
except Exception as e:
    print(f"Failed to export to ONNX: {str(e)}\n{traceback.format_exc()}")

# Evaluate
try:
    results = model.val(save=True, save_txt=True, conf=0.5, iou=0.45)
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    for i, ap in enumerate(results.box.ap50):
        print(f"Class {results.names[i]} AP@0.5: {ap:.4f}")
    results.confusion_matrix.plot(save_dir="wood_defect_runs")
except Exception as e:
    print(f"Validation failed: {str(e)}\n{traceback.format_exc()}")

# Predict on test images
if test_dir and os.path.exists(test_dir):
    test_images = glob.glob(f"{test_dir}/*.[jpb][pnm][gfp]")
    if test_images:
        try:
            for img in test_images[:3]:
                results = model.predict(img, conf=0.5, iou=0.45, save=True)
                results[0].show()
        except Exception as e:
            print(f"Prediction failed: {str(e)}\n{traceback.format_exc()}")
    else:
        print(f"No test images found in {test_dir}")
else:
    print(f"Test directory not specified in dataset.yaml or does not exist")