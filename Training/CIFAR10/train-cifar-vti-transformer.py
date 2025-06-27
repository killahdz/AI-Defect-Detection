import tensorflow as tf
from transformers import ViTImageProcessor, TFViTForImageClassification
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt

# 🚀 Set seeds
tf.random.set_seed(42)
np.random.seed(42)

# 📥 Load CIFAR-10 from Hugging Face Datasets
ds = load_dataset("cifar10")
train = ds["train"]
test = ds["test"]

# 🛠 Build processor and model
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
IMG_SIZE = processor.size["height"]  # e.g. 224
NUM_LABELS = 10

model = TFViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=NUM_LABELS
)

# 🔄 Preprocessing function
def preprocess(examples):
    images = [img.convert("RGB") if hasattr(img, "convert") else img for img in examples["img"]]
    proc = processor(images=images, return_tensors="np")
    proc["labels"] = examples["label"]
    return proc

# 🧹 Clean Dataset
train = train.map(preprocess, batched=True, remove_columns=["img"])
test = test.map(preprocess, batched=True, remove_columns=["img"])

# 🎯 Convert to tf.data.Dataset
def to_tf(ds_split):
    return tf.data.Dataset.from_tensor_slices({
        "pixel_values": ds_split["pixel_values"],
        "labels": ds_split["labels"]
    })

train_ds = to_tf(train)
test_ds = to_tf(test)

# 📦 Pipeline: resize, batch, shuffle
BATCH = 32
AUTO = tf.data.AUTOTUNE

def prepare(batch):
    x = tf.image.resize(batch["pixel_values"], (IMG_SIZE, IMG_SIZE))
    return {"pixel_values": x}, batch["labels"]

train_ds = train_ds.shuffle(len(train_ds)).map(prepare, num_parallel_calls=AUTO).batch(BATCH).prefetch(AUTO)
test_ds = test_ds.map(prepare, num_parallel_calls=AUTO).batch(BATCH).prefetch(AUTO)

# ⚙️ Compile and train
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

early = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(train_ds, validation_data=test_ds, epochs=10, callbacks=[early])

# 📈 Evaluate
loss, acc = model.evaluate(test_ds)
print(f"Test loss: {loss:.4f}, accuracy: {acc:.4f}")

# 📊 Plot training curves
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epoch"), plt.ylabel("Accuracy"), plt.legend()
plt.show()

# 💾 Export for inference
export_path = "c:/training/vit_cifar10_trained"

@tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32, name="pixel_values")])
def serve(pixel_values):
    outputs = model(pixel_values, training=False)
    return {"output_0": outputs.logits}

tf.saved_model.save(model, export_path, signatures={"serve": serve})
print("✅ Model saved to", export_path)
