from flask import Flask, request, jsonify, render_template
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# 載入 .keras 格式的模型
model = load_model("cnn_model.keras")

# 引入特徵萃取函式（需自行實作）
from feature_extraction import extract_features

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "sequence" not in data:
        return jsonify({"error": "Missing 'sequence' in request"}), 400

    sequence = data["sequence"]
    
    try:
        features = extract_features(sequence)  # shape: (L, D)
        features = np.expand_dims(features, axis=0)  # shape: (1, L, D)
        pred = model.predict(features)[0][0]
        result = "SNARE" if pred >= 0.5 else "Not SNARE"
        return jsonify({"result": result, "score": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
