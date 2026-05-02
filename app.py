from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import tensorflow as tf
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# 🔥 FIX CORS (IMPORTANT)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model
MODEL_PATH = "model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded")
except Exception as e:
    print("❌ Model load error:", e)
    model = None


@app.route('/')
def home():
    return "Backend running 🚀"


def predict_image(img):
    h, w, _ = img.shape

    patch_size = 64
    step = 32

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


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        stats = predict_image(img)

        _, buffer = cv2.imencode('.jpg', img)
        encoded = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "image": encoded,
            "stats": stats
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500


# 🔥 Render fix
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)