
from __future__ import division, print_function
import os
import numpy as np

# TensorFlow / Keras
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "Model.keras")
FALLBACK_MODEL_PATH = "Model.hdf5"
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 38


def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    model = Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), padding="valid", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, (11, 11), strides=(1, 1), padding="valid", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(384, (3, 3), strides=(1, 1), padding="valid", activation="relu"))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(384, (3, 3), strides=(1, 1), padding="valid", activation="relu"))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding="valid", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1000, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model

# Load your trained model
print(" ** Model Loading **")
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, compile=False)
else:
    model = build_model()
    model.load_weights(FALLBACK_MODEL_PATH)
print(" ** Model Loaded **")



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = x/255

    preds = model.predict(x, verbose=0)
    class_index = int(np.argmax(preds, axis=1)[0])
    li = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    class_name = li[class_index].split('___')
    return class_name


@app.route('/')
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        class_name = model_predict(file_path, model)

        result = str(f"Predicted Crop:{class_name[0]}  Predicted Disease:{class_name[1].title().replace('_',' ')}")               
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
