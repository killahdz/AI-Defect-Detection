from ultralytics import YOLO
import os

# Load pretrained model
model = YOLO("yolov8s.pt")  # "yolov8s.pt" for better accuracy

# Train
model.train(
    data="C:/training/kaggle/dataset.yaml",
    epochs=50,
    imgsz=128,
    batch=64,
    name="wood_defect_model",  # Custom name for the training run
    save=True,  # Ensure checkpoints are saved
    save_period=10,  # Save checkpoint every 10 epochs,
    augment=True
)

# Save the model in PyTorch format (optional, for backup)
pt_save_path = "C:/training/kaggle/models/wood_defect_yolov8s.pt"
os.makedirs(os.path.dirname(pt_save_path), exist_ok=True)
model.save(pt_save_path)
print(f"PyTorch model saved to {pt_save_path}")

# Export the model to ONNX format
onnx_save_path = "C:/training/kaggle/models/wood_defect_yolov8s.onnx"
try:
    model.export(format="onnx", imgsz=128, dynamic=True)  # Export with dynamic batch size
    print(f"ONNX model saved to {onnx_save_path}")
except Exception as e:
    print(f"Failed to export to ONNX: {str(e)}")

# Evaluate on validation set
results = model.val()
print(results)

# Predict on a sample image
image_path = None
for ext in [".jpg", ".png", ".jpeg"]:
    potential_path = f"C:/training/kaggle/Images - 1/99100004{ext}"
    if os.path.exists(potential_path):
        image_path = potential_path
        break

if image_path:
    results = model.predict(image_path)
    results[0].show()  # Display results with bounding boxes
else:
    print("Sample image not found: tried extensions .jpg, .png, .jpeg for 99100004")