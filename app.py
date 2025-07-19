from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import joblib
import shap

app = Flask(__name__)

# Load models and expected features
models = joblib.load("best_xgb_models.pkl")
expected_features = joblib.load("xgb_expected_features.pkl")

# Modern UI template
UPLOAD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>UTI Antibiotic Resistance Predictor</title>
  <style>
    body {
      font-family: 'Segoe UI', sans-serif;
      background-color: #f0f2f5;
      margin: 0;
      padding: 40px;
      display: flex;
      justify-content: center;
    }
    .container {
      background-color: #fff;
      border-radius: 12px;
      box-shadow: 0 6px 24px rgba(0,0,0,0.08);
      padding: 40px;
      max-width: 600px;
      width: 100%;
    }
    h2 {
      text-align: center;
      color: #1a1a1a;
    }
    form {
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 10px;
      margin-bottom: 30px;
    }
    input[type="file"] {
      font-size: 14px;
    }
    button {
      background-color: #007bff;
      color: white;
      padding: 10px 20px;
      font-weight: 600;
      border: none;
      border-radius: 6px;
      cursor: pointer;
    }
    .results {
      margin-top: 30px;
    }
    .pill {
      display: inline-block;
      padding: 12px 20px;
      border-radius: 50px;
      font-weight: 600;
      font-size: 16px;
      margin: 10px 0;
      color: white;
    }
    .resistant-pill {
      background-color: #e74c3c;
    }
    .sensitive-pill {
      background-color: #27ae60;
    }
    .ab-label {
      font-weight: bold;
      margin-right: 10px;
    }
    .card {
      background-color: white;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      padding: 20px;
      margin-top: 30px;
      text-align: left;
    }
    .highlight {
      font-weight: bold;
      font-size: 17px;
      margin-bottom: 10px;
    }
    ol {
      padding-left: 20px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>UTI Antibiotic Resistance Predictor</h2>
    <form id="upload-form">
      <input type="file" id="csv-file" name="file" accept=".csv" required>
      <button type="submit">Predict</button>
    </form>
    <div id="results" class="results"></div>
  </div>

  <script>
    document.getElementById("upload-form").addEventListener("submit", async function(event) {
      event.preventDefault();
      const file = document.getElementById("csv-file").files[0];
      if (!file) {
        alert("Please upload a CSV file.");
        return;
      }
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/upload", {
        method: "POST",
        body: formData
      });

      const html = await response.text();
      document.getElementById("results").innerHTML = html;
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

        # Predict and gather results
        for label, model in models.items():
            predictions[label] = int(model.predict(df)[0])
            probabilities[label] = float(model.predict_proba(df)[0][1])

            # Generate SHAP explanation for resistant cases
            if predictions[label] == 1:
                explainer = shap.Explainer(model)
                shap_values = explainer(df)

                top_shap_df = pd.DataFrame({
                    "feature": df.columns,
                    "shap_value": shap_values.values[0]
                }).sort_values(by="shap_value", key=abs, ascending=False).head(5)

                html_block = f"<div class='card'><div class='highlight'>{label} - Top 5 Contributing Features</div><ol>"
                for _, row in top_shap_df.iterrows():
                    html_block += f"<li>{row['feature']} (Impact: {row['shap_value']:.4f})</li>"
                html_block += "</ol></div>"

                shap_sections.append(html_block)

        # Generate pill-style output
        results_html = "<div><h3 style='text-align:left;'>Prediction Results</h3>"
        for ab in predictions:
            prob_percent = f"{probabilities[ab]*100:.0f}%"
            status = "Resistant" if predictions[ab] == 1 else "Sensitive"
            pill_class = "resistant-pill" if predictions[ab] == 1 else "sensitive-pill"
            results_html += f"<div class='pill {pill_class}'><span class='ab-label'>{ab}</span> {status} ({prob_percent})</div><br>"
        results_html += "</div>"

        # Add SHAP summary cards
        shap_html = "".join(shap_sections)

        return results_html + shap_html

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
