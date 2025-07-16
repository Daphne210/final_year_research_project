from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("xgb_baseline_model.pkl")  # update with your actual model file path

# Define expected features
expected_features = [
    "age", "gender", "bacteria", "urine_ph", "prior_antibiotic_use",  # example features
    # Add all features your model was trained on
    # test comments image
]

@app.route('/', methods=['GET'])
def home():
    return "🧪 AMR Prediction API is running. Use POST /predict with JSON input.", 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json(force=True)

        # Validate and extract features
        if not all(feature in input_data for feature in expected_features):
            return jsonify({
                "error": "Missing required features",
                "expected_features": expected_features
            }), 400

        # Create DataFrame for model input
        input_df = pd.DataFrame([input_data])[expected_features]

        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df).tolist()

        return jsonify({
            "prediction": int(prediction[0]),
            "probability": probability[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
