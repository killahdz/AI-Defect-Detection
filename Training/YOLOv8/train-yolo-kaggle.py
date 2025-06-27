from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolov8n.pt")  # Nano model for faster training; use "yolov8m.pt" for better accuracy

# Train
model.train(data="C:/training/kaggle/dataset.yaml", epochs=50, imgsz=128, batch=32)

# Evaluate
results = model.val()
print(results)

# Predict on a sample image
results = model.predict("C:/training/kaggle/Images - 1/99100004.jpg")  # Adjust extension if needed
results[0].show()  # Display results with bounding boxes


#Input Image (3x128x128)
#  ↓
#[Backbone]
#  Conv (16x64x64) → C2f → Conv (32x32x32) → C2f → Conv (64x16x16) → C2f
#  ↓
#  Conv (128x8x8) → C2f → Conv (256x4x4) → C2f → SPPF (256x4x4)
#  ↓
#[Neck]
#  Upsample (256x8x8) → Concat (384x8x8) → C2f (128x8x8)
#  ↓
#  Upsample (128x16x16) → Concat (192x16x16) → C2f (64x16x16)
#  ↓
#  Conv (64x8x8) → Concat (192x8x8) → C2f (128x8x8)
#  ↓
#  Conv (128x4x4) → Concat (384x4x4) → C2f (256x4x4)
#  ↓
#[Head]
#  Detect: Predicts bounding boxes, classes, and confidence scores
#  ↓
#Output: List of [x, y, w, h, confidence, class_probabilities] for each detected object