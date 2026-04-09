import json
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Train a modern TF/Keras model on the combined dataset.
# Usage: python train_model.py
# Optional env vars:
#   DATA_DIR   (default: "disease data/combined")
#   EPOCHS     (default: 10)
#   BATCH_SIZE (default: 32)
#   MODEL_OUT  (default: "Model.keras")
#   WEIGHTS_OUT (default: "Model.weights.h5")
#   IMG_SIZE   (default: 224)

DATA_DIR = Path(os.environ.get("DATA_DIR", "disease data/combined"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
MODEL_OUT = os.environ.get("MODEL_OUT", "Model.keras")
WEIGHTS_OUT = os.environ.get("WEIGHTS_OUT", "Model.weights.h5")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))

train_dir = DATA_DIR / "train"
valid_dir = DATA_DIR / "valid"

if not train_dir.exists() or not valid_dir.exists():
    raise SystemExit("Expected train/valid folders in DATA_DIR")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode="nearest",
)
valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)
valid_set = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

class_names = list(training_set.class_indices.keys())
with open("class_names.json", "w", encoding="utf-8") as handle:
    json.dump(class_names, handle, indent=2)

num_classes = len(class_names)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

model = Sequential(
    [
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    training_set,
    validation_data=valid_set,
    epochs=EPOCHS,
)

model.save(MODEL_OUT)
model.save_weights(WEIGHTS_OUT)

print("Saved model:", MODEL_OUT)
print("Saved weights:", WEIGHTS_OUT)
