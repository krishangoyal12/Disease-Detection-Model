import os
import tensorflow as tf

# Run this script in a Python 3.8/3.9 env with TensorFlow 2.10/2.12.
# It converts the legacy HDF5 model to the modern Keras format.

src = os.environ.get("SRC_MODEL", "Model.hdf5")
dst = os.environ.get("DST_MODEL", "Model.keras")

print("Loading:", src)
model = tf.keras.models.load_model(src, compile=False)
print("Saving:", dst)
model.save(dst)
print("Done")
