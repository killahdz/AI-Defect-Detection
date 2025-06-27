import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Image input size (adjust as needed)
img_height, img_width = 128, 128
batch_size = 32
data_dir = "C:/training/kaggle"

# L2 regularization
l2_reg = regularizers.l2(1e-4)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'
)

# Get class count
num_classes = len(train_generator.class_indices)

# Build CNN model
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
