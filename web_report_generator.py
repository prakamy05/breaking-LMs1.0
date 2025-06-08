import os
import pandas as pd
from jinja2 import Template

# Paths
REPORT_DIR = "toxicity_analysis_individual_and_combined"
SUMMARY_CSV = os.path.join(REPORT_DIR, "toxicity_metrics_summary.csv")
OUTPUT_HTML = os.path.join(REPORT_DIR, "toxicity_report.html")

# Load metrics summary
df = pd.read_csv(SUMMARY_CSV)

# Sort: individual first, then combined files
combined_tags = ["combined_all_files", "combined_catalytic_files", "combined_non_catalytic_files"]
df["IsCombined"] = df["Tag"].apply(lambda t: t in combined_tags)
df_sorted = pd.concat([
    df[~df["IsCombined"]],
    df[df["IsCombined"]]
])

# Build HTML using Jinja2 template
template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Toxicity Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 30px; }
        h1, h2 { color: #2c3e50; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
        th, td { padding: 8px 12px; border: 1px solid #ccc; text-align: center; }
        th { background-color: #f2f2f2; }
        img { max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ccc; }
        .section { margin-bottom: 50px; }
    </style>
</head>
<body>
    <h1>🧪 Toxicity Analysis Report</h1>
    <p><b>Generated from:</b> {{ summary_csv }}</p>

    {% for row in rows %}
    <div class="section">
        <h2>📂 {{ row.Tag }}</h2>
        <table>
            <tr>
                <th>Samples</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Accuracy</th>
            </tr>
            <tr>
                <td>{{ row['Total Samples'] }}</td>
                <td>{{ row.Precision }}</td>
                <td>{{ row.Recall }}</td>
                <td>{{ row['F1 Score'] }}</td>
                <td>{{ row.Accuracy }}</td>
            </tr>
        </table>
        <h4>Confusion Matrix</h4>
        <img src="{{ row.Tag }}_confusion_matrix.png" alt="Confusion Matrix">
        <h4>Toxicity Score Plot</h4>
        <img src="{{ row.Tag }}_toxicity_scores.png" alt="Toxicity Score Plot">
    </div>
    {% endfor %}
</body>
</html>
""")

# Render HTML
html = template.render(
    summary_csv=os.path.basename(SUMMARY_CSV),
    rows=df_sorted.to_dict(orient="records")
)

# Save
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ HTML report saved to: {OUTPUT_HTML}")
