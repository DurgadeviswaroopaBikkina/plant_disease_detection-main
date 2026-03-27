import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import cv2


# ===============================
# PyInstaller-safe resource path
# ===============================
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS   # PyInstaller temp folder
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ===============================
# Flask App Configuration
# ===============================
app = Flask(
    __name__,
    template_folder=resource_path('templates'),
    static_folder=resource_path('static')
)

UPLOAD_FOLDER = resource_path('uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ===============================
# Load Model
# ===============================
model_path = resource_path('model.h5')
model = load_model(model_path, compile=False)

print("Model loaded successfully.")
print("Open http://127.0.0.1:5000/ in your browser")


# ===============================
# Labels
# ===============================
labels = {
    0: 'Healthy',
    1: 'Powdery',
    2: 'Rust'
}


# ===============================
# Prediction Function
# ===============================
def getResult(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions


# ===============================
# Routes
# ===============================
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    predictions = getResult(file_path)
    predicted_label = labels[np.argmax(predictions)]

    return predicted_label


# ===============================
# Run App
# ===============================
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
