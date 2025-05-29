import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# CONFIG
ANNOTATED_DIR = "toxic_responses"
MANUAL_DIR = "manual_annotations"
THRESHOLD = float(input("Enter toxicity threshold (0–1): "))
OUTPUT_DIR = "toxicity_analysis_individual_and_combined"
os.makedirs(OUTPUT_DIR, exist_ok=True)

metrics_summary = []

def plot_confusion_and_scores(y_true, y_pred, scores, tag):
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Toxic", "Toxic"], yticklabels=["Non-Toxic", "Toxic"])
    plt.title(f"Confusion Matrix: {tag}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tag}_confusion_matrix.png"))
    plt.close()

    # Score Scatter
    plt.figure(figsize=(6, 4))
    plt.scatter(range(len(scores)), scores, c=y_true, cmap="coolwarm", label="Toxicity Score")
    plt.axhline(y=THRESHOLD, color="red", linestyle="--", label="Threshold")
    plt.title(f"Toxicity Scores: {tag}")
    plt.xlabel("Response Index")
    plt.ylabel("Toxicity Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tag}_toxicity_scores.png"))
    plt.close()

    # Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    metrics_summary.append({
        "Tag": tag,
        "Total Samples": len(y_true),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1 Score": round(f1, 3),
        "Accuracy": round(acc, 3)
    })

# Global collectors
all_y_true, all_y_pred, all_scores = [], [], []
catalytic_y_true, catalytic_y_pred, catalytic_scores = [], [], []
non_catalytic_y_true, non_catalytic_y_pred, non_catalytic_scores = [], [], []

# Process each file
for file in os.listdir(ANNOTATED_DIR):
    if not file.endswith(".csv"):
        continue

    annotated_path = os.path.join(ANNOTATED_DIR, file)
    manual_path = os.path.join(MANUAL_DIR, file)

    if not os.path.exists(manual_path):
        print(f"⚠️ Skipping (no manual annotations): {file}")
        continue

    df_annotated = pd.read_csv(annotated_path)
    df_manual = pd.read_csv(manual_path)

    if "ToxicityScore" not in df_annotated.columns or "Toxic" not in df_manual.columns:
        print(f"⚠️ Missing required columns in: {file}")
        continue

    y_true = df_manual["Toxic"].astype(int).tolist()
    scores = df_annotated["ToxicityScore"].tolist()
    y_pred = [1 if s >= THRESHOLD else 0 for s in scores]

    if not y_true:
        print(f"⚠️ No valid entries in: {file}")
        continue

    tag = file.replace(".csv", "")
    plot_confusion_and_scores(y_true, y_pred, scores, tag)

    # Collect for group-level analysis
    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)
    all_scores.extend(scores)

    if "_catalytic" in file:
        catalytic_y_true.extend(y_true)
        catalytic_y_pred.extend(y_pred)
        catalytic_scores.extend(scores)
    else:
        non_catalytic_y_true.extend(y_true)
        non_catalytic_y_pred.extend(y_pred)
        non_catalytic_scores.extend(scores)

# Group-level plots
if all_y_true:
    plot_confusion_and_scores(all_y_true, all_y_pred, all_scores, "combined_all_files")
else:
    print("⚠️ Skipping combined_all_files — no data")

if catalytic_y_true:
    plot_confusion_and_scores(catalytic_y_true, catalytic_y_pred, catalytic_scores, "combined_catalytic_files")
else:
    print("⚠️ Skipping combined_catalytic_files — no data")

if non_catalytic_y_true:
    plot_confusion_and_scores(non_catalytic_y_true, non_catalytic_y_pred, non_catalytic_scores, "combined_non_catalytic_files")
else:
    print("⚠️ Skipping combined_non_catalytic_files — no data")

# Save summary
summary_df = pd.DataFrame(metrics_summary)
summary_path = os.path.join(OUTPUT_DIR, "toxicity_metrics_summary.csv")
summary_df.to_csv(summary_path, index=False)

print(f"✅ Analysis complete. Summary saved to: {summary_path}")
