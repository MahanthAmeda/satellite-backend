from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
import base64
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# ✅ FIX CORS COMPLETELY
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load model safely
MODEL_PATH = "model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded")
except Exception as e:
    print("❌ Model load error:", e)
    model = None


# ✅ Test route
@app.route("/")
def home():
    return "🚀 Backend is running"


# ✅ Prediction function (SAFE)
def predict_image(img):
    try:
        img = cv2.resize(img, (128, 128))
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        prediction = model.predict(img, verbose=0)[0]

        return {
            "urban": float(round(prediction[0] * 100, 2)),
            "vegetation": float(round(prediction[1] * 100, 2)),
            "water": float(round(prediction[2] * 100, 2)),
        }

    except Exception as e:
        print("Prediction error:", e)
        return {"urban": 0.0, "vegetation": 0.0, "water": 0.0}


# ✅ Analyze route (NO CRASH VERSION)
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        stats = predict_image(img)

        # send original image back
        _, buffer = cv2.imencode(".jpg", img)
        encoded = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "image": encoded,
            "stats": stats
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ✅ Render port fix
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)