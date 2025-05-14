import os
# Suppress TensorFlow warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR

from flask import Flask, request, send_file, jsonify
import numpy as np
from PIL import Image
import io
import joblib
import tensorflow as tf
from keras.models import load_model
from keras.saving import register_keras_serializable

app = Flask(__name__)

# Define custom loss function
mse = tf.keras.losses.MeanSquaredError()

@register_keras_serializable()
def generator_loss(fake_output, real_output):
    real_output = tf.cast(real_output, 'float32')
    return mse(fake_output, real_output)

# Load the model ONCE at startup
try:
    model = joblib.load("model.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

IMG_SIZE = 128  # Based on your model input

# Preprocess input grayscale image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_np = np.array(image).astype(np.float32) / 255.0  # Normalize to [0,1]
    image_np = np.expand_dims(image_np, axis=(0, -1))  # Shape: (1, 128, 128, 1)
    return image_np

# Postprocess output image (RGB)
def postprocess_image(output_array):
    output_array = np.clip(output_array[0], 0, 1)  # Shape: (128, 128, 3)
    output_image = Image.fromarray((output_array * 255).astype(np.uint8))
    return output_image

# Health check route
@app.route('/')
def health_check():
    return "✅ Server is running", 200

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    input_array = preprocess_image(image_bytes)

    try:
        output_array = model.predict(input_array)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    output_image = postprocess_image(output_array)

    img_io = io.BytesIO()
    output_image.save(img_io, format='PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

# Run app on dynamic port for Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # 10000 for local fallback
    app.run(host='0.0.0.0', port=port)
