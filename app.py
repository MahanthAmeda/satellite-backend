from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import tensorflow as tf
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)

# Load model
MODEL_PATH = "model.keras"

if not os.path.exists(MODEL_PATH):
    print("❌ model not found")
    model = None
else:
    print("✅ Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded")


# 🧠 Prediction (batch + weighted)
def predict_image(img):
    h, w, _ = img.shape

    patch_size = 64
    step = 32

    classes = ["urban", "vegetation", "water"]

    patches = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            patch = img[y:y+patch_size, x:x+patch_size]

            if patch.size == 0:
                continue

            patch = cv2.resize(patch, (128, 128))
            patches.append(patch)

    if len(patches) == 0:
        return {"urban": 0.0, "vegetation": 0.0, "water": 0.0}

    patches = np.array(patches, dtype=np.float32)
    patches = preprocess_input(patches)

    predictions = model.predict(patches, verbose=0)

    scores = {"urban": 0.0, "vegetation": 0.0, "water": 0.0}

    for pred in predictions:
        scores["urban"] += float(pred[0])
        scores["vegetation"] += float(pred[1])
        scores["water"] += float(pred[2])

    total = sum(scores.values())

    return {
        "urban": float(round(scores["urban"] / total * 100, 2)),
        "vegetation": float(round(scores["vegetation"] / total * 100, 2)),
        "water": float(round(scores["water"] / total * 100, 2)),
    }


# 🎨 Segmentation
def generate_segmentation(img):
    h, w, _ = img.shape

    patch_size = 64
    step = 32

    classes = ["urban", "vegetation", "water"]

    patches = []
    coords = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            patch = img[y:y+patch_size, x:x+patch_size]

            if patch.size == 0:
                continue

            patch_resized = cv2.resize(patch, (128, 128))
            patches.append(patch_resized)
            coords.append((y, x))

    if len(patches) == 0:
        return img

    patches = np.array(patches, dtype=np.float32)
    patches = preprocess_input(patches)

    predictions = model.predict(patches, verbose=0)

    overlay = img.copy()

    for i, pred in enumerate(predictions):
        y, x = coords[i]
        label = classes[int(np.argmax(pred))]

        if label == "water":
            color = (255, 0, 0)
        elif label == "vegetation":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        y_end = min(y + patch_size, h)
        x_end = min(x + patch_size, w)

        overlay[y:y_end, x:x_end] = (
            0.6 * overlay[y:y_end, x:x_end] +
            0.4 * np.array(color)
        )

    return overlay.astype(np.uint8)


# 🌐 API
@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    # 🔥 Validate file
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "Empty file"}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Process
    stats = predict_image(img)
    segmented_img = generate_segmentation(img)

    _, buffer = cv2.imencode('.jpg', segmented_img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "image": encoded_image,
        "stats": stats
    })


if __name__ == '__main__':
    app.run(debug=True)