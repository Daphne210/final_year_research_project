from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# Load all antibiotic models and expected feature list
models = joblib.load("best_xgb_models.pkl") 
expected_features = joblib.load("xgb_expected_features.pkl") 

# HTML for file upload UI
UPLOAD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AMR Prediction - Upload CSV</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 30px; }
    h2 { color: #333; }
    .container { max-width: 600px; }
    .button { padding: 10px 15px; margin-top: 10px; }
    #results { margin-top: 30px; }
  </style>
</head>
<body>
  <div class="container">
    <h2> AMR Prediction - Upload CSV</h2>
    <form id="upload-form">
      <input type="file" id="csv-file" name="file" accept=".csv" required>
      <br><br>
      <button type="submit" class="button">Make Predictions</button>
    </form>

    <div id="results"></div>
  </div>

  <script>
    document.getElementById("upload-form").addEventListener("submit", async function(event) {
      event.preventDefault();
      const fileInput = document.getElementById("csv-file");
      const file = fileInput.files[0];

      if (!file) {
        alert("Please select a CSV file.");
        return;
      }

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/upload", {
        method: "POST",
        body: formData
      });

      const text = await response.text();
      document.getElementById("results").innerHTML = response.ok ? text : `<pre>${text}</pre>`;
    });
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(UPLOAD_HTML)

@app.route("/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    try:
        df = pd.read_csv(file)

        missing = [f for f in expected_features if f not in df.columns]
        if missing:
            return jsonify({"error": "Missing columns", "missing": missing}), 400

        df = df[expected_features]

        # Predict for each antibiotic using its respective model
        for label, model in models.items():
            df[f"{label}_prediction"] = model.predict(df)

        return df.to_html(classes="table table-striped", index=False)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict_json():
    try:
        input_data = request.get_json(force=True)

        if not all(feature in input_data for feature in expected_features):
            return jsonify({
                "error": "Missing required features",
                "expected_features": expected_features
            }), 400

        input_df = pd.DataFrame([input_data])[expected_features]

        predictions_dict = {
            label: int(model.predict(input_df)[0])
            for label, model in models.items()
        }

        return jsonify(predictions_dict)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
