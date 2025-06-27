import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load and normalize CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalize to [0,1]

# CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# L2 regularization factor
l2_reg = regularizers.l2(1e-4)

# Build CNN model
model = models.Sequential([
    # Block 1
    layers.Conv2D(64, (3, 3), padding='same', activation='relu',kernel_regularizer=l2_reg, input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu',kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # Block 2
    layers.Conv2D(128, (3, 3), padding='same', activation='relu',kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu',kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # Block 3
    layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # Dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10)  # Output logits (no softmax, applied in loss)
])

# Compile model with label smoothing
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(train_images)

# Training callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

# Train the model
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=64),
    epochs=50,
    validation_data=(test_images, test_labels),
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
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

model_name = "cifarhyper"

plt.tight_layout()
os.makedirs('c:/training', exist_ok=True)
plt.savefig(f'c:/training/cifar_training_{model_name}_plots.png')
plt.show()

# ==== Save Model with Consistent Inference Signature ====

export_path = f'c:/training/{model_name}.model'

@tf.function(input_signature=[tf.TensorSpec([None, 32, 32, 3], tf.float32, name='keras_tensor')])
def serving_fn(keras_tensor):
    logits = model(keras_tensor, training=False)
    return {'output_0': logits}

tf.saved_model.save(model, export_path, signatures={'serve': serving_fn})
print(f"Model saved to {export_path} with explicit 'serve' signature.")
