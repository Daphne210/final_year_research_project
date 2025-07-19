from flask import Flask, request, jsonify, render_template_string, send_file
import pandas as pd
import joblib
import shap
import os
import matplotlib.pyplot as plt
import uuid
from fpdf import FPDF

app = Flask(__name__)

# Load models and expected features
models = joblib.load("best_xgb_models.pkl")
expected_features = joblib.load("xgb_expected_features.pkl")

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

UPLOAD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>UTI Antibiotic Resistance Predictor</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background-color: #f4f6f9;
      padding: 40px;
      display: flex;
      justify-content: center;
    }
    .card {
      background-color: white;
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      padding: 30px;
      max-width: 700px;
      width: 100%;
    }
    h1 {
      text-align: center;
      color: #111;
    }
    .predict-button {
      background-color: #007bff;
      color: white;
      border: none;
      padding: 10px 16px;
      font-size: 16px;
      border-radius: 6px;
      cursor: pointer;
    }
    .result-row {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      margin-top: 20px;
    }
    .pill {
      flex: 0 0 48%;
      margin-bottom: 15px;
      padding: 12px;
      color: white;
      font-weight: bold;
      border-radius: 20px;
      text-align: center;
    }
    .resistant {
      background-color: #e74c3c;
    }
    .sensitive {
      background-color: #2ecc71;
    }
    .section {
      margin-top: 30px;
    }
    .shap-img {
      max-width: 100%;
      margin-top: 10px;
    }
    .btn-group {
      margin-top: 20px;
      display: flex;
      gap: 20px;
    }
    .btn-group a {
      background: #007bff;
      color: white;
      padding: 8px 14px;
      border-radius: 6px;
      text-decoration: none;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>UTI Antimicrobial Resistance Predictor</h1>
    <form id="upload-form">
      <input type="file" id="csv-file" name="file" accept=".csv" required>
      <button type="submit" class="predict-button">Predict</button>
    </form>
    <div id="results" class="section"></div>
  </div>

  <script>
    document.getElementById("upload-form").addEventListener("submit", async function(event) {
      event.preventDefault();
      const file = document.getElementById("csv-file").files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/upload", {
        method: "POST",
        body: formData
      });

      const html = await response.text();
      document.getElementById("results").innerHTML = response.ok ? html : `<pre>${html}</pre>`;
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
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        # Keep only expected features
        df = df[[col for col in df.columns if col in expected_features]]
        df = df[expected_features]

        predictions = {}
        probabilities = {}
        explanations = []
        csv_rows = []

        uid = str(uuid.uuid4())[:8]

        for abx, model in models.items():
            pred = int(model.predict(df)[0])
            prob = float(model.predict_proba(df)[0][1])
            predictions[abx] = pred
            probabilities[abx] = prob

            csv_rows.append([abx, "Resistant" if pred else "Sensitive", f"{prob*100:.2f}%"])

            if pred == 1:
                explainer = shap.Explainer(model)
                shap_vals = explainer(df)
                top_df = pd.DataFrame({
                    "feature": df.columns,
                    "shap": shap_vals.values[0]
                }).sort_values(by="shap", key=abs, ascending=False).head(5)

                # Plotting
                plt.figure(figsize=(6, 3))
                plt.barh(top_df["feature"], top_df["shap"], color="skyblue")
                plt.gca().invert_yaxis()
                plt.title(f"Top 5 Features for {abx} Resistance")
                shap_path = f"static/shap_{uid}_{abx}.png"
                plt.tight_layout()
                plt.savefig(shap_path)
                plt.close()

                explanations.append(f"""
                    <div class='section'>
                        <h4>{abx} - Top 5 Contributing Features</h4>
                        <ol>
                            {''.join(f"<li>{row['feature']} (Impact: {row['shap']:.4f})</li>" for _, row in top_df.iterrows())}
                        </ol>
                        <img src='/{shap_path}' class='shap-img'>
                    </div>
                """)

        # Clinical suggestion
        tips = ""
        if any(predictions.values()):
            tips = "<ul>"
            for abx in predictions:
                if predictions[abx] == 1:
                    tips += f"<li>Avoid prescribing <b>{abx}</b>. Consider alternative therapy.</li>"
            tips += "</ul>"
        else:
            tips = "All antibiotics show low resistance risk. Proceed with standard empiric treatment."

        # Save CSV
        csv_file = f"static/report_{uid}.csv"
        pd.DataFrame(csv_rows, columns=["Antibiotic", "Prediction", "Probability"]).to_csv(csv_file, index=False)

        # Save PDF
        pdf_path = f"static/report_{uid}.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="UTI Antimicrobial Resistance Report", ln=True, align='C')
        pdf.ln(10)
        for abx, label, prob in csv_rows:
            pdf.cell(200, 10, txt=f"{abx}: {label} ({prob})", ln=True)
        pdf.output(pdf_path)

        # Build prediction pills
        pill_html = ""
        for abx in predictions:
            label = "Resistant" if predictions[abx] == 1 else "susceptible"
            color = "resistant" if predictions[abx] == 1 else "susceptible"
            percent = f"{probabilities[abx]*100:.0f}%"
            pill_html += f"<div class='pill {color}'>{abx}&nbsp;&nbsp;{label} ({percent})</div>"

        # Assemble final HTML
        return f"""
            <div class='result-row'>{pill_html}</div>
            <div class='section'>
                <h3>Clinical Suggestion</h3>
                {tips}
            </div>
            {''.join(explanations)}
            <div class='btn-group'>
                <a href='/{csv_file}' download>Download CSV</a>
                <a href='/{pdf_path}' download>Download PDF</a>
            </div>
        """

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
