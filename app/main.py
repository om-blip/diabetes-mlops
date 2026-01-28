from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/model.pkl")

@app.route("/")
def home():
    return {"message": "Diabetes Prediction API"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return {"prediction": int(prediction)}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
