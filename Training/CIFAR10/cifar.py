import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
import matplotlib.pyplot as plt

# Load and normalize CIFAR-10 test data
(_, _), (test_images, test_labels) = datasets.cifar10.load_data()
test_images = test_images / 255.0  # Normalize as done during training

# Define class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the SavedModel
loaded_model = tf.saved_model.load("c:/training/cifar.model")

# Get the serving signature
infer = loaded_model.signatures['serve']

# Prepare input data
input_data = tf.convert_to_tensor(test_images, dtype=tf.float32)

# Make predictions for all images in a single batch
predictions = infer(keras_tensor=input_data)['output_0']

# Convert predictions to probabilities (since model uses from_logits=True)
probabilities = tf.nn.softmax(predictions)

# Get predicted classes
predicted_classes = tf.argmax(probabilities, axis=1)

# Evaluate accuracy
accuracy = tf.keras.metrics.sparse_categorical_accuracy(test_labels, predictions)
mean_accuracy = tf.reduce_mean(accuracy).numpy()
print(f"Test accuracy: {mean_accuracy}")

# Evaluate loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = loss_fn(test_labels, predictions).numpy()
print(f"Test loss: {loss}")

# Set up interactive Matplotlib figure
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(5, 5))
img_display = ax.imshow(test_images[0])  # Initialize with first image
title = ax.set_title("")
ax.axis('off')

# Iterate through all images and update display
for i in range(len(test_images)):
    # Update image
    img_display.set_data(test_images[i])
    
    # Update title with true label, predicted label, and confidence
    true_label = class_names[test_labels[i][0]]
    pred_label = class_names[predicted_classes[i].numpy()]
    confidence = probabilities[i][predicted_classes[i]].numpy() * 100
    title.set_text(f"Image {i+1}/{len(test_images)}\nTrue: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)")
    
    # Redraw the figure
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # Pause briefly to allow viewing (0.5 seconds)
    plt.pause(0.5)

    # Optional: Uncomment to wait for key press instead of auto-advance
    # input("Press Enter to show next image...")

plt.ioff()  # Turn off interactive mode
plt.close()
print("Finished displaying all images.")