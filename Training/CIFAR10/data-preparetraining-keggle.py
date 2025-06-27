import os
import shutil
from sklearn.model_selection import train_test_split

image_dir = "C:/training/kaggle/Images - 1"
annotation_dir = "C:/training/kaggle/Bounding Boxes - YOLO Format - 1"
train_dir = "C:/training/kaggle/train/images"
train_ann_dir = "C:/training/kaggle/train/labels"
val_dir = "C:/training/kaggle/val/images"
val_ann_dir = "C:/training/kaggle/val/labels"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(train_ann_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(val_ann_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

for f in train_files:
    shutil.copy(os.path.join(image_dir, f), os.path.join(train_dir, f))
    ann_f = f.rsplit(".", 1)[0] + ".txt"
    if os.path.exists(os.path.join(annotation_dir, ann_f)):
        shutil.copy(os.path.join(annotation_dir, ann_f), os.path.join(train_ann_dir, ann_f))

for f in val_files:
    shutil.copy(os.path.join(image_dir, f), os.path.join(val_dir, f))
    ann_f = f.rsplit(".", 1)[0] + ".txt"
    if os.path.exists(os.path.join(annotation_dir, ann_f)):
        shutil.copy(os.path.join(annotation_dir, ann_f), os.path.join(val_ann_dir, ann_f))