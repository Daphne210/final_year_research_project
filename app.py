# app.py (Complete Final Version)

from flask import Flask, request, jsonify, render_template_string, send_file
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import uuid
import os
from fpdf import FPDF

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
      max-width: 700px;
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
      margin-bottom: 20px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      justify-items: center;
    }
    .pill {
      display: inline-block;
      padding: 12px 20px;
      border-radius: 50px;
      font-weight: 600;
      font-size: 16px;
      color: white;
    }
    .resistant-pill { background-color: #e74c3c; }
    .sensitive-pill { background-color: #27ae60; }
    .ab-label { font-weight: bold; margin-right: 10px; }
    .card {
      background-color: white;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      padding: 20px;
      margin-top: 20px;
    }
    .btn-group {
      text-align: right;
      margin-top: 20px;
    }
    .btn-group a {
      text-decoration: none;
      padding: 10px 15px;
      background-color: #007bff;
      color: white;
      border-radius: 5px;
      margin-left: 10px;
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
    <div id="results"></div>
  </div>
  <script>
    document.getElementById("upload-form").addEventListener("submit", async function(event) {
      event.preventDefault();
      const file = document.getElementById("csv-file").files[0];
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
        df = df[expected_features]

        predictions, probabilities, shap_sections = {}, {}, []
        decision_suggestions = []
        session_id = str(uuid.uuid4())
        shap_plot_paths = []

        for label, model in models.items():
            pred = int(model.predict(df)[0])
            prob = float(model.predict_proba(df)[0][1])
            predictions[label] = pred
            probabilities[label] = prob

            if pred == 1:
                explainer = shap.Explainer(model)
                shap_values = explainer(df)

                shap_df = pd.DataFrame({
                    "feature": df.columns,
                    "shap_value": shap_values.values[0]
                }).sort_values(by="shap_value", key=abs, ascending=False).head(5)

                suggestion = f"Avoid using {label}. Consider alternative antibiotic."
                decision_suggestions.append(suggestion)

                # SHAP force plot
                plt.figure()
                shap.plots.bar(shap_values[0], show=False)
                path = f"static/shap_{session_id}_{label}.png"
                plt.savefig(path, bbox_inches='tight')
                shap_plot_paths.append((label, path))

                explanation = f"<div class='card'><div class='highlight'>{label} - Top 5 Contributing Features</div><ol>"
                for _, row in shap_df.iterrows():
                    explanation += f"<li>{row['feature']} (Impact: {row['shap_value']:.4f})</li>"
                explanation += f"</ol><img src='/{path}' width='100%'></div>"
                shap_sections.append(explanation)

        # Pills
        results_html = "<div><h3 style='text-align:left;'>Prediction Results</h3><div class='grid'>"
        for ab in predictions:
            p = f"{probabilities[ab]*100:.0f}%"
            status = "Resistant" if predictions[ab] == 1 else "Sensitive"
            pill_class = "resistant-pill" if predictions[ab] == 1 else "sensitive-pill"
            results_html += f"<div class='pill {pill_class}'><span class='ab-label'>{ab}</span> {status} ({p})</div>"
        results_html += "</div></div>"

        # Decision suggestions
        suggestion_block = "<div class='card'><strong>Clinical Decision Suggestions:</strong><ul>"
        if decision_suggestions:
            for tip in decision_suggestions:
                suggestion_block += f"<li>{tip}</li>"
        else:
            suggestion_block += "<li>All antibiotics predicted sensitive. Proceed with standard treatment.</li>"
        suggestion_block += "</ul></div>"

        # CSV and PDF download links
        result_df = pd.DataFrame({
            "Antibiotic": list(predictions.keys()),
            "Prediction": ["Resistant" if predictions[k] == 1 else "Sensitive" for k in predictions],
            "Probability": [round(probabilities[k], 4) for k in probabilities]
        })
        csv_path = f"static/report_{session_id}.csv"
        pdf_path = f"static/report_{session_id}.pdf"
        result_df.to_csv(csv_path, index=False)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="AMR Prediction Report", ln=True, align='C')
        for idx, row in result_df.iterrows():
            pdf.cell(200, 10, txt=f"{row['Antibiotic']}: {row['Prediction']} ({row['Probability']*100:.1f}%)", ln=True)
        pdf.output(pdf_path)

        download_links = f"""
        <div class='btn-group'>
          <a href='/{csv_path}' download>⬇️ CSV</a>
          <a href='/{pdf_path}' download>⬇️ PDF</a>
        </div>
        """

        return results_html + suggestion_block + ''.join(shap_sections) + download_links

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
