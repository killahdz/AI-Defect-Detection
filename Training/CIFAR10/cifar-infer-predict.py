import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
import matplotlib.pyplot as plt

# Load and normalize CIFAR-10 test data
(_, _), (test_images, test_labels) = datasets.cifar10.load_data()
test_images = test_images / 255.0  # Normalize as done during training

# Define class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model_name = "cifarhyper"

# Load the SavedModel
loaded_model = tf.saved_model.load(f"c:/training/{model_name}.model")

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

# Initialize counters for correct and incorrect predictions per class
num_classes = len(class_names)
correct_counts = np.zeros(num_classes, dtype=int)
incorrect_counts = np.zeros(num_classes, dtype=int)

# Set up interactive Matplotlib figures
plt.ion()  # Turn on interactive mode

# Figure 1: Image viewer
fig1, ax1 = plt.subplots(figsize=(5, 5))
img_display = ax1.imshow(test_images[0])  # Initialize with first image
title = ax1.set_title("")
ax1.axis('off')

# Figure 2: Tally grid (table)
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.axis('off')  # Hide axes for table
table_data = [[class_names[i], 0, 0, 0.0] for i in range(num_classes)] + [['Total', 0, 0, 0.0]]
table = ax2.table(cellText=table_data,
                  colLabels=['Class', 'Correct', 'Incorrect', 'Success Rate (%)'],
                  loc='center',
                  cellLoc='center',
                  colWidths=[0.3, 0.2, 0.2, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)  # Adjust table size

# Handle division by zero for success rate
np.seterr(all='ignore')  # Suppress warnings for 0/0 cases

# Iterate through all images
total_correct = 0
total_incorrect = 0
for i in range(len(test_images)):
    # Check if either figure is closed
    if not plt.get_fignums():
        print(f"Stopped at image {i+1} due to window close.")
        break

    # Update image viewer
    img_display.set_data(test_images[i])
    true_label = class_names[test_labels[i][0]]
    pred_label = class_names[predicted_classes[i].numpy()]
    confidence = probabilities[i][predicted_classes[i]].numpy() * 100
    title.set_text(f"Image {i+1}/{len(test_images)}\nTrue: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)")
    fig1.canvas.draw()
    fig1.canvas.flush_events()

    # Update counters
    true_class = test_labels[i][0]
    pred_class = predicted_classes[i].numpy()
    if true_class == pred_class:
        correct_counts[true_class] += 1
        total_correct += 1
    else:
        incorrect_counts[true_class] += 1
        total_incorrect += 1

    # Update tally grid with success rates
    for j in range(num_classes):
        table.get_celld()[(j+1, 1)].get_text().set_text(str(correct_counts[j]))
        table.get_celld()[(j+1, 2)].get_text().set_text(str(incorrect_counts[j]))
        total_j = correct_counts[j] + incorrect_counts[j]
        success_rate = (correct_counts[j] / total_j * 100) if total_j > 0 else 0.0
        table.get_celld()[(j+1, 3)].get_text().set_text(f"{success_rate:.1f}")
    table.get_celld()[(num_classes+1, 1)].get_text().set_text(str(total_correct))
    table.get_celld()[(num_classes+1, 2)].get_text().set_text(str(total_incorrect))
    total_all = total_correct + total_incorrect
    total_success_rate = (total_correct / total_all * 100) if total_all > 0 else 0.0
    table.get_celld()[(num_classes+1, 3)].get_text().set_text(f"{total_success_rate:.1f}")
    fig2.canvas.draw()
    fig2.canvas.flush_events()

    # Pause briefly to allow viewing (0.5 seconds)
    plt.pause(0.1)

    # Optional: Uncomment to wait for key press instead of auto-advance
    # input("Press Enter to show next image...")

# Final output
plt.ioff()  # Turn off interactive mode
plt.close('all')
print("Finished displaying images." if i == len(test_images)-1 else f"Stopped early at image {i+1}.")
print("\nFinal Tally:")
for i in range(num_classes):
    total_i = correct_counts[i] + incorrect_counts[i]
    success_rate = (correct_counts[i] / total_i * 100) if total_i > 0 else 0.0
    print(f"{class_names[i]}: Correct={correct_counts[i]}, Incorrect={incorrect_counts[i]}, Success Rate={success_rate:.1f}%")
print(f"Total: Correct={total_correct}, Incorrect={total_incorrect}, Success Rate={total_success_rate:.1f}%")