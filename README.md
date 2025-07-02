# AI Defect Detection – YOLOv8 Training & Evaluation Suite

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![MIT License](https://img.shields.io/github/license/killahdz/AI-Defect-Detection)
![Last Commit](https://img.shields.io/github/last-commit/killahdz/AI-Defect-Detection)

This repository contains professional-grade Python scripts for training, evaluating, and visualizing YOLOv8-based object detection models, originally built for wood defect identification but adaptable to other domains.

![Sample Output](https://github.com/user-attachments/assets/140c93aa-e7e8-4a5d-bcdb-40cfa0352265)

---

## 🎯 Purpose

Developed to support high-accuracy AI defect detection in timber processing and manufacturing, this toolkit is suitable for any object detection task requiring:

- Model training with configurable augmentation
- Evaluation with bounding box matching and heatmap visualization
- Conversion and export of models (PyTorch → ONNX)
- Real-time overlay and diagnostics using OpenCV and Matplotlib

---

## 🚀 Quickstart

1. **Clone the repository**:
   ```sh
   git clone https://github.com/killahdz/AI-Defect-Detection.git
   cd AI-Defect-Detection
   ```

2. **Install dependencies**:
   ```sh
   pip install ultralytics opencv-python matplotlib numpy scipy seaborn pyyaml torch
   ```

3. **Prepare your dataset** (see below).

4. **Train a model**:
   ```sh
   python train_yolov8.py --data dataset.yaml --model yolov8m.pt --img 1280 --epochs 100
   ```

5. **Evaluate your trained model**:
   ```sh
   python eval_yolov8.py --weights wood_defect_runs/best.pt --data dataset.yaml
   ```

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

## 📂 Directory Layout

| Path                         | Description                                    |
|------------------------------|------------------------------------------------|
| `train_yolov8.py`            | Training script for YOLOv8                     |
| `eval_yolov8.py`             | Evaluation and visualization script            |
| `utils.py`                   | Utility functions (augmentation, metrics, etc) |
| `models/*.pt` / `*.onnx`     | Trained model weights (YOLOv8 + exported ONNX) |
| `dataset.yaml`               | Dataset configuration for YOLOv8 training      |
| `train/images`               | Training images                                |
| `train/labels`               | YOLO-format label files                        |
| `wood_defect_runs/`          | Output directory for model checkpoints/plots   |

---

## 📑 Dataset Format

- Images are stored in `train/images/`
- Labels (YOLO format) in `train/labels/`
- `dataset.yaml` example:
  ```yaml
  train: ./train/images
  val: ./val/images
  nc: 4  # number of classes
  names: ['knot', 'crack', 'hole', 'discoloration']
  ```

---

## ⚙️ Model Variants (YOLOv8)

| Model         | Parameters | Use Case                              |
|---------------|------------|----------------------------------------|
| `yolov8n.pt`  | ~3M        | Lightweight (Jetson, edge inference)   |
| `yolov8s.pt`  | ~11M       | Balanced (fast + accurate)             |
| `yolov8m.pt`  | ~25M       | Higher accuracy, mid-range GPU         |
| `yolov8l.pt`  | ~43M       | High accuracy, heavy GPU (A100 etc.)   |
| `yolov8x.pt`  | ~68M       | Maximum accuracy, slowest              |

**Recommended:** `yolov8m.pt` with 1280×1280 or `yolov8s.pt` with 960×960 for balanced speed and accuracy.

---

## 🛠️ Requirements

- Python 3.10+
- `ultralytics`
- `torch`
- `opencv-python`
- `matplotlib`
- `numpy`
- `scipy`
- `seaborn`
- `pyyaml`

Install via pip:
```sh
pip install ultralytics opencv-python matplotlib numpy scipy seaborn pyyaml torch
```

---

## 🚀 Example Flow

1. **Train** a YOLOv8 model on wood defects using a custom `.yaml`
2. **Evaluate** predictions across images and track per-class metrics
3. **Visualize** overlayed detections and confusion heatmaps
4. **Export** to `.onnx` and use in downstream inference pipelines

---

## 🧩 Configuration & Options

- All scripts accept standard CLI arguments (`--data`, `--weights`, `--img`, `--epochs`, etc.).
- Augmentation and validation options can be toggled in the script/config.
- Results and logs are saved in `wood_defect_runs/`.

---

## 👨‍💻 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.  
If you use this toolkit in your research or production, a citation or mention is appreciated.

---

## 🧠 Built For

- Computer vision researchers
- AI engineers working on visual QA
- Industrial automation teams using camera-based defect detection
- Students or practitioners studying YOLOv8 workflows

---

## 📄 License

This repository is licensed under the [MIT License](./LICENSE).  
© 2025 Daniel Kereama

---

## 👨‍💻 About the Author

**Daniel Kereama** is a senior engineer with 20+ years of experience in enterprise .NET, computer vision, and applied AI.  
Focused on building production-grade, diagnostics-friendly tooling for model evaluation in industrial AI deployments.

---

## 📬 Contact

- GitHub: [github.com/killahdz](https://github.com/killahdz)
- LinkedIn: [linkedin.com/in/danielkereama](https://linkedin.com/in/danielkereama)
