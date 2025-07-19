from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import joblib
import shap

app = Flask(__name__)

models = joblib.load("best_xgb_models.pkl")
expected_features = joblib.load("xgb_expected_features.pkl")

UPLOAD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>UTI Antibiotic Resistance Predictor</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background-color: #f4f6f8;
      padding: 40px;
      text-align: center;
    }
    .container {
      max-width: 800px;
      margin: auto;
    }
    .card {
      background-color: white;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      padding: 20px;
      margin-top: 30px;
      text-align: left;
    }
    h2 {
      margin-bottom: 20px;
    }
    button {
      background-color: #007bff;
      color: white;
      padding: 10px 20px;
      border: none;
      border-radius: 6px;
      cursor: pointer;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }
    th, td {
      padding: 8px 12px;
      border-bottom: 1px solid #eee;
      text-align: left;
    }
    .resistant {
      color: #c0392b;
      font-weight: bold;
    }
    .sensitive {
      color: #27ae60;
      font-weight: bold;
    }
    .highlight {
      font-weight: bold;
      margin-bottom: 10px;
    }
    ul {
      margin-top: 10px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>UTI Antibiotic Resistance Predictor</h2>
    <form id="upload-form">
      <input type="file" id="csv-file" name="file" accept=".csv" required>
      <br><br>
      <button type="submit">Predict</button>
    </form>
    <div id="results"></div>
  </div>

  <script>
    document.getElementById("upload-form").addEventListener("submit", async function(event) {
      event.preventDefault();
      const fileInput = document.getElementById("csv-file");
      const file = fileInput.files[0];
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

      const resultHtml = await response.text();
      document.getElementById("results").innerHTML = resultHtml;
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

                html_block = f"<div class='card'><div class='highlight'>{label} - Top 5 Contributing Features</div><ol>"
                for _, row in top_shap_df.iterrows():
                    html_block += f"<li>{row['feature']} (Impact: {row['shap_value']:.4f})</li>"
                html_block += "</ol></div>"

                shap_sections.append(html_block)

        # Format main prediction summary
        results_html = "<div class='card'><div class='highlight'>Prediction Results</div><table><thead><tr><th>Antibiotic</th><th>Prediction</th><th>Probability</th></tr></thead><tbody>"
        for ab in predictions:
            status = "Resistant" if predictions[ab] == 1 else "Sensitive"
            style = "resistant" if predictions[ab] == 1 else "sensitive"
            prob = f"{probabilities[ab]:.3f}"
            results_html += f"<tr><td>{ab}</td><td class='{style}'>{status}</td><td>{prob}</td></tr>"
        results_html += "</tbody></table></div>"

        # Combine and return full result
        shap_html = "".join(shap_sections)
        return results_html + shap_html

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
