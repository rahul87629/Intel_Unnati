import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# === Model and Image Folder Paths ===
MODEL_PATH = "mobilenet_thermo_model.h5"  # Replace with actual model path
IMAGE_FOLDER = "REAL_TIME"

# === Load model ===
model = tf.keras.models.load_model(MODEL_PATH)

# === Predict Function ===
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    # Convert raw label to user-friendly label
    raw_label = "Control" if prediction > 0.5 else "DM"
    friendly_label = "Non-Diabetes" if raw_label == "Control" else "Diabetes"

    return friendly_label, prediction

# === Loop and Predict ===
for img_file in sorted(os.listdir(IMAGE_FOLDER)):
    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(IMAGE_FOLDER, img_file)
        label, confidence = predict_image(img_path)
        print(f"{img_file} ➜ Predicted: {label} (confidence: {confidence:.4f})")
