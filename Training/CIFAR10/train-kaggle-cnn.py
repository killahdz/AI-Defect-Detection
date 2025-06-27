import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from collections import Counter

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Image input size
img_height, img_width = 128, 128
batch_size = 32
data_dir = "C:/training/kaggle"
image_dir = os.path.join(data_dir, "Images - 1")
annotation_dir = os.path.join(data_dir, "Bounding Boxes - YOLO Format - 1")

# L2 regularization
l2_reg = regularizers.l2(1e-4)

# Get all class IDs from annotations
def get_class_ids(annotation_dir):
    class_ids = set()
    for ann_file in os.listdir(annotation_dir):
        if not ann_file.endswith(".txt"):
            continue
        ann_path = os.path.join(annotation_dir, ann_file)
        try:
            with open(ann_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.strip():
                    try:
                        print(line);
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
                    except (IndexError, ValueError):
                        print(f"Invalid line in {ann_file}: {line.strip()}")
        except Exception as e:
            print(f"Error reading {ann_file}: {str(e)}")
    class_ids = sorted(class_ids)
    print(f"Detected class IDs: {class_ids}")
    return class_ids

# Define class names
class_ids = get_class_ids(annotation_dir)
class_names = {i: f"defect_{i}" for i in class_ids}  # Dynamic mapping
class_names.update({0: "no_defect", 2: "knot", 4: "scratch", 7: "unknown_defect"})  # Based on examples

print(f"Detected class IDs: {class_ids}")
print(f"Class names mapping: {class_names}")

# Custom data generator
class WoodDefectGenerator(Sequence):
    def __init__(self, image_dir, annotation_dir, class_names, target_size=(128, 128), batch_size=32, subset="training", validation_split=0.2, datagen=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.class_names = class_names
        self.target_size = target_size
        self.batch_size = batch_size
        self.subset = subset
        self.validation_split = validation_split
        self.datagen = datagen
        
        # Get list of images
        image_extensions = (".jpg", ".jpeg", ".png")
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
        self.labels = []
        
        # Assign labels based on annotations
        for img_file in self.image_files:
            ann_file = img_file.rsplit(".", 1)[0] + ".txt"
            ann_path = os.path.join(annotation_dir, ann_file)
            if os.path.exists(ann_path):
                with open(ann_path, "r") as f:
                    lines = f.readlines()
                class_ids = [int(line.split()[0]) for line in lines if line.strip()]
                # Use most frequent class ID
                class_id = max(set(class_ids), key=class_ids.count) if class_ids else 0
            else:
                class_id = 0  # Default to no_defect if no annotation
            self.labels.append(class_id)
        
        # Map class IDs to indices
        self.class_indices = {name: idx for idx, name in enumerate(sorted(class_names.values()))}
        self.labels = [self.class_indices[class_names.get(cid, f"defect_{cid}")] for cid in self.labels]
        
        # Split into training and validation
        self.indices = np.arange(len(self.image_files))
        np.random.shuffle(self.indices)
        split_idx = int(len(self.image_files) * (1 - validation_split))
        if subset == "training":
            self.indices = self.indices[:split_idx]
        else:
            self.indices = self.indices[split_idx:]
        
        print(f"{subset.capitalize()} set: {len(self.indices)} images")
    
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = []
        
        for i in batch_indices:
            img_file = self.image_files[i]
            img_path = os.path.join(self.image_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            img = cv2.resize(img, self.target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = img / 255.0  # Normalize
            
            if self.datagen:
                img = self.datagen.random_transform(img)
            
            batch_images.append(img)
            batch_labels.append(self.labels[i])
        
        return np.array(batch_images), np.array(batch_labels)
    
    def get_class_indices(self):
        return self.class_indices

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Create generators
train_generator = WoodDefectGenerator(
    image_dir=image_dir,
    annotation_dir=annotation_dir,
    class_names=class_names,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset="training",
    validation_split=0.2,
    datagen=train_datagen
)

val_generator = WoodDefectGenerator(
    image_dir=image_dir,
    annotation_dir=annotation_dir,
    class_names=class_names,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset="validation",
    validation_split=0.2,
    datagen=None  # No augmentation for validation
)

# Get class count
num_classes = len(train_generator.get_class_indices())
print(f"Number of classes: {num_classes}")

# Build CNN model (unchanged from your original script)
model = models.Sequential([
    layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg, input_shape=(img_height, img_width, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes)
])

# Compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# Training callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(val_generator, verbose=2)
print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Plot training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

model_name = "kaggle_wood_defect"
plt.tight_layout()
os.makedirs('c:/training', exist_ok=True)
plt.savefig(f'c:/training/training_{model_name}_plots.png')
plt.show()

# Save model with a TF serving signature
export_path = f'c:/training/{model_name}.model'

@tf.function(input_signature=[tf.TensorSpec([None, img_height, img_width, 3], tf.float32, name='keras_tensor')])
def serving_fn(keras_tensor):
    logits = model(keras_tensor, training=False)
    return {'output_0': logits}

tf.saved_model.save(model, export_path, signatures={'serve': serving_fn})
print(f"Model saved to {export_path} with explicit 'serve' signature.")