from flask import Flask, request, jsonify, render_template
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Conv1D, MaxPooling1D, Flatten, Dense

app = Flask(__name__)

# ✅ 重建模型架構
def build_model():
    model = Sequential([
        InputLayer(batch_input_shape=(None, 500, 20), name="input_layer"),
        Conv1D(filters=64, kernel_size=3, activation='relu', name="conv1d"),
        MaxPooling1D(pool_size=2, name="max_pooling1d"),
        Flatten(name="flatten"),
        Dense(64, activation='relu', name="dense"),
        Dense(1, activation='sigmoid', name="dense_1")
    ])
    return model

# ✅ 載入模型權量
model = build_model()
model.load_weights("cnn_model.h5")

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
        result = "SNARE" if pred >= 0.61 else "Not SNARE"
        return jsonify({"result": result, "score": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
