from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the serialized model pipeline
model = joblib.load("superkart_model.joblib")


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "SuperKart Sales Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    """Single prediction from a JSON payload."""
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return jsonify({"predicted_sales": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Batch prediction from an uploaded CSV file."""
    try:
        file = request.files["file"]
        df = pd.read_csv(file)
        predictions = model.predict(df)
        df["Predicted_Sales"] = predictions.round(2)
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
