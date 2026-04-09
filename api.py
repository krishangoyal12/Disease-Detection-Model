import json
import os
from io import BytesIO
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers

MODEL_PATH = os.environ.get("MODEL_PATH", "Model.keras")
FALLBACK_MODEL_PATH = "Model.hdf5"
INPUT_SHAPE = (224, 224, 3)
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", "class_names.json")

# Keep the class list aligned with training_set.class_indices order.
DEFAULT_CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as handle:
        CLASS_NAMES = json.load(handle)
else:
    CLASS_NAMES = DEFAULT_CLASS_NAMES

NUM_CLASSES = len(CLASS_NAMES)


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


app = FastAPI()
model = None


@app.on_event("startup")
def load_model_on_startup():
    global model
    print(" ** Model Loading **")
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, compile=False)
    else:
        model = build_model()
        model.load_weights(FALLBACK_MODEL_PATH)
    print(" ** Model Loaded **")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        img = image.load_img(BytesIO(contents), target_size=(224, 224))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    preds = model.predict(x, verbose=0)
    class_index = int(np.argmax(preds, axis=1)[0])
    class_name = CLASS_NAMES[class_index]
    crop, disease = class_name.split("___")

    return JSONResponse(
        {
            "crop": crop,
            "disease": disease.replace("_", " "),
            "class": class_name,
            "confidence": float(np.max(preds)),
        }
    )
