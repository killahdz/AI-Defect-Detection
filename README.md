# AI Defect Detection – YOLOv8 Training & Evaluation Suite

This repository contains a set of professional-grade Python scripts for training, evaluating, and visualizing YOLOv8-based object detection models, originally built for wood defect identification. It is designed for industrial AI workflows and includes robust support for both custom training and real-time inference analysis.

![image](https://github.com/user-attachments/assets/140c93aa-e7e8-4a5d-bcdb-40cfa0352265)

## 🎯 Purpose

Developed to support high-accuracy AI defect detection in timber processing and manufacturing, this toolkit is suitable for any object detection task requiring:

- Model training with configurable augmentation
- Evaluation with bounding box matching and heatmap visualization
- Conversion and export of models (PyTorch → ONNX)
- Real-time overlay and diagnostics using OpenCV and Matplotlib

---

## 📦 What’s Included

### ✅ Training Scripts
Custom YOLOv8 training routines with advanced configuration:

- Supports multiple model sizes (`n`, `s`, `m`, `l`, `x`)
- Augmentation for color, brightness, mosaic, and rotation
- Automatic saving of models to `.pt` and `.onnx`
- GPU auto-detection and AMP support
- Optional validation + visualization of results
- Configurable dataset format (Kaggle-style or Dataset Ninja)

### ✅ Evaluation Scripts
Interactive tools for inference and metrics:

- Live prediction with YOLOv8 checkpoints
- Bounding box matching via **Hungarian Algorithm**
- Per-class **precision**, **recall**, and **F1-score** tracking
- Heatmap generation of class-wise performance over time
- Non-Max Suppression and annotation validation logic

---

## 🔍 Sample Use Cases

- 🔧 Fine-tune YOLOv8 models on custom wood defect datasets
- 🧪 Evaluate model generalization across varying class distributions
- 📊 Visualize performance for each class as images stream through
- 🚀 Export trained weights for production deployment or edge use (ONNX-ready)

---

## 📂 Directory Layout

| Path                         | Description                                                   |
|------------------------------|---------------------------------------------------------------|
| `*.py`                       | Training or evaluation scripts                                |
| `models/*.pt` / `*.onnx`     | Trained model weights (YOLOv8 + exported ONNX)               |
| `dataset.yaml`               | Dataset configuration for YOLOv8 training                    |
| `train/images`               | Training images                                               |
| `train/labels`               | YOLO-format label files                                       |
| `wood_defect_runs/`          | Output directory for model checkpoints and plots             |

---

## ⚙️ Model Variants (YOLOv8)

| Model         | Parameters | Use Case                              |
|---------------|------------|----------------------------------------|
| `yolov8n.pt`  | ~3M        | Lightweight (Jetson, edge inference)   |
| `yolov8s.pt`  | ~11M       | Balanced (fast + accurate)             |
| `yolov8m.pt`  | ~25M       | Higher accuracy, mid-range GPU         |
| `yolov8l.pt`  | ~43M       | High accuracy, heavy GPU (A100 etc.)   |
| `yolov8x.pt`  | ~68M       | Maximum accuracy, slowest              |

**Recommended:** `yolov8m.pt` with 1280×1280 or `yolov8s.pt` with 960×960 for balance of speed and accuracy.

---

## 🛠️ Requirements

Install dependencies:



- Python 3.10+
- `ultralytics`
- `pytorch`
- `opencv-python`
- `matplotlib`
- `numpy`
- `scipy`
- `seaborn`
- `pyyaml`

Install via pip:

`pip install ultralytics opencv-python matplotlib numpy scipy seaborn pyyaml pytorch`

---

## 🚀 Example Flow

1. **Train** a YOLOv8 model on wood defects using a custom `.yaml`
2. **Evaluate** predictions across images and track per-class metrics
3. **Visualize** overlayed detections and confusion heatmaps
4. **Export** to `.onnx` and use in downstream inference pipelines

---

## 🧠 Built For

- Computer vision researchers
- AI engineers working on visual QA
- Industrial automation teams using camera-based defect detection
- Students or practitioners studying YOLOv8 workflows

## 📄 License

This repository is licensed under the [MIT License](./LICENSE).  
© 2025 Daniel Kereama

---

## 👨‍💻 About the Author

**Daniel Kereama** is a senior engineer with 20+ years of experience in enterprise .NET, computer vision, and applied AI.  
Built with a focus on building production-grade, diagnostics-friendly tooling for model evaluation in industrial AI deployments.

---

## 📬 Contact

- GitHub: [github.com/killahdz](https://github.com/killahdz)
- LinkedIn: [linkedin.com/in/danielkereama](https://linkedin.com/in/danielkereama)
