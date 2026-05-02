from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def home():
    return "🚀 Backend is running (light mode)"


# 🧠 Lightweight prediction (no TensorFlow)
def predict_image(img):
    # Resize to reduce processing
    img = cv2.resize(img, (128, 128))

    # Convert to float
    img = img.astype(np.float32)

    # Compute average colors
    b_mean = np.mean(img[:, :, 0])
    g_mean = np.mean(img[:, :, 1])
    r_mean = np.mean(img[:, :, 2])

    total = b_mean + g_mean + r_mean + 1e-6

    # Simple heuristic
    water = b_mean / total
    vegetation = g_mean / total
    urban = r_mean / total

    return {
        "urban": float(round(urban * 100, 2)),
        "vegetation": float(round(vegetation * 100, 2)),
        "water": float(round(water * 100, 2)),
    }


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        stats = predict_image(img)

        # Return original image
        _, buffer = cv2.imencode(".jpg", img)
        encoded = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "image": encoded,
            "stats": stats
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500


# Render port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)