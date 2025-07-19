from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import joblib
import shap

app = Flask(__name__)

# Load models and expected features
models = joblib.load("best_xgb_models.pkl")
expected_features = joblib.load("xgb_expected_features.pkl")

UPLOAD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AMR Prediction - Upload CSV</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 30px; background-color: #f9f9f9; }
    h2 { color: #333; }
    .container { max-width: 600px; }
    .button { padding: 10px 15px; margin-top: 10px; }
    #results { margin-top: 30px; }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
    th, td { padding: 10px; text-align: center; border: 1px solid #ccc; }
  </style>
</head>
<body>
  <div class="container">
    <h2>🧪 AMR Prediction - Upload CSV</h2>
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
        df.columns = df.columns.str.strip()
        df = df[[col for col in df.columns if col in expected_features]]
        missing = [f for f in expected_features if f not in df.columns]
        if missing:
            return jsonify({"error": "Missing columns", "missing": missing}), 400
        df = df[expected_features]

        predictions = {}
        probabilities = {}
        shap_sections = []

        for label, model in models.items():
            predictions[label] = int(model.predict(df)[0])
            probabilities[label] = float(model.predict_proba(df)[0][1])

            if predictions[label] == 1:
                explainer = shap.Explainer(model)
                shap_values = explainer(df)

                top_shap_df = pd.DataFrame({
                    "feature": df.columns,
                    "shap_value": shap_values.values[0]
                }).sort_values(by="shap_value", key=abs, ascending=False).head(5)

                html_block = f"<h3>Top 5 Features Contributing to {label} Resistance</h3><ol>"
                for _, row in top_shap_df.iterrows():
                    html_block += f"<li><strong>{row['feature']}</strong>: {row['shap_value']:.4f}</li>"
                html_block += "</ol>"

                shap_sections.append(html_block)

        # Build summary and prediction tables
        resistant = [ab for ab, pred in predictions.items() if pred == 1]
        if resistant:
            summary = f"⚠️ Patient is likely resistant to: <strong>{', '.join(resistant)}</strong>."
        else:
            summary = "✅ No resistance detected. All antibiotics are likely effective."

        pred_html = "<h3>Prediction Results</h3><table><tr>"
        for ab in predictions:
            pred_html += f"<th>{ab}</th>"
        pred_html += "</tr><tr>"
        for ab, pred in predictions.items():
            color = "#e74c3c" if pred == 1 else "#2ecc71"
            label = "Resistant" if pred == 1 else "Susceptible"
            pred_html += f"<td style='background-color:{color};color:white;font-weight:bold'>{label}</td>"
        pred_html += "</tr></table>"

        prob_html = "<h3>Prediction Probabilities</h3><ul>"
        if resistant:
            for ab in resistant:
                prob_html += f"<li><strong>{ab}</strong>: {probabilities[ab]*100:.2f}%</li>"
        else:
            for ab, prob in probabilities.items():
                prob_html += f"<li><strong>{ab}</strong>: {prob*100:.2f}%</li>"
        prob_html += "</ul>"

        shap_html = "".join(shap_sections)

        return f"""
            <div style='font-family:Arial'>
                {pred_html}
                <p style='font-size:16px;margin-top:20px'>{summary}</p>
                {prob_html}
                {shap_html}
            </div>
        """

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
