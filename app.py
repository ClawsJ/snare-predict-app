from flask import Flask, request, jsonify, render_template
import numpy as np
from keras.models import load_model

app = Flask(__name__)
model = load_model("cnn_model.h5")

# 假設你已經有一個函數可以從序列轉成 CNN 特徵（如 PSSM、one-hot）
from feature_extraction import extract_features

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sequence = data["sequence"]
    
    features = extract_features(sequence)  # shape (1, L, D)
    pred = model.predict(np.array([features]))[0][0]

    result = "SNARE" if pred >= 0.5 else "Not SNARE"
    return jsonify({"result": result, "score": float(pred)})
