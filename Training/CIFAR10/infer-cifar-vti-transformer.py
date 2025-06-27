import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from transformers import ViTImageProcessor

# Load and normalize CIFAR-10 test data
(_, _), (test_images, test_labels) = datasets.cifar10.load_data()
test_images = test_images.astype(np.uint8)  # Keep original pixel range 0–255

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load processor and SavedModel
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
loaded_model = tf.saved_model.load("c:/training/vit_cifar10_trained")
infer = loaded_model.signatures['serve']
IMG_SIZE = processor.size['height']  # e.g., 224

# Preprocess all test images: resize, normalize, batch
resized = tf.image.resize(test_images, [IMG_SIZE, IMG_SIZE])
input_tensor = processor(images=resized.numpy(), return_tensors="tf")["pixel_values"]

# Run inference
predictions = infer(pixel_values=input_tensor)["output_0"]
probabilities = tf.nn.softmax(predictions)
predicted_classes = tf.argmax(probabilities, axis=1)

# Calculate overall accuracy and loss
accuracy = tf.keras.metrics.sparse_categorical_accuracy(test_labels, predictions)
mean_accuracy = tf.reduce_mean(accuracy).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = loss_fn(test_labels, predictions).numpy()

print(f"Test accuracy: {mean_accuracy:.4f}")
print(f"Test loss: {loss:.4f}")

# Prepare interactive display components
num_classes = len(class_names)
correct_counts = np.zeros(num_classes, dtype=int)
incorrect_counts = np.zeros(num_classes, dtype=int)

plt.ion()
fig1, ax1 = plt.subplots(figsize=(5, 5))
img_plot = ax1.imshow(test_images[0])
title = ax1.set_title("", fontsize=14)
ax1.axis("off")

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.axis("off")
table_data = [[class_names[i], 0, 0, 0.0] for i in range(num_classes)] + [["Total", 0, 0, 0.0]]
table = ax2.table(cellText=table_data,
                  colLabels=["Class", "Correct", "Incorrect", "Success Rate (%)"],
                  loc="center", cellLoc="center", colWidths=[0.3,0.2,0.2,0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1,1.5)
np.seterr(all="ignore")

total_correct = 0
total_incorrect = 0

# Display predictions one image at a time
for i in range(len(test_images)):
    if not plt.get_fignums(): break

    img_plot.set_data(test_images[i])
    true_label = class_names[test_labels[i][0]]
    pred_label = class_names[predicted_classes[i].numpy()]
    confidence = probabilities[i][predicted_classes[i]].numpy() * 100
    true_c = test_labels[i][0]
    pred_c = predicted_classes[i].numpy()

    symbol = "✅" if true_c == pred_c else "❌"
    title_color = "green" if true_c == pred_c else "red"
    title.set_text(f"{symbol} Image {i+1}/{len(test_images)}\nTrue: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)")
    title.set_color(title_color)

    fig1.canvas.draw()
    fig1.canvas.flush_events()

    if true_c == pred_c:
        correct_counts[true_c] += 1
        total_correct += 1
    else:
        incorrect_counts[true_c] += 1
        total_incorrect += 1

    for j in range(num_classes):
        table.get_celld()[(j+1,1)].get_text().set_text(str(correct_counts[j]))
        table.get_celld()[(j+1,2)].get_text().set_text(str(incorrect_counts[j]))
        total_j = correct_counts[j] + incorrect_counts[j]
        table.get_celld()[(j+1,3)].get_text().set_text(f"{(correct_counts[j]/total_j*100) if total_j else 0.0:.1f}")
    table.get_celld()[(num_classes+1,1)].get_text().set_text(str(total_correct))
    table.get_celld()[(num_classes+1,2)].get_text().set_text(str(total_incorrect))
    total_rate = (total_correct / (total_correct + total_incorrect) * 100) if (total_correct + total_incorrect) else 0.0
    table.get_celld()[(num_classes+1,3)].get_text().set_text(f"{total_rate:.1f}")

    fig2.canvas.draw()
    fig2.canvas.flush_events()
    plt.pause(0.1)

plt.ioff()
plt.close("all")
print("Finished displaying images." if i == len(test_images)-1 else f"Stopped at image {i+1}.")
print("\nFinal Tally:")
for i in range(num_classes):
    total_i = correct_counts[i] + incorrect_counts[i]
    print(f"{class_names[i]}: Correct={correct_counts[i]}, Incorrect={incorrect_counts[i]}, Success Rate={(correct_counts[i]/total_i*100 if total_i else 0.0):.1f}%")
print(f"Total: Correct={total_correct}, Incorrect={total_incorrect}, Success Rate={total_rate:.1f}%")
