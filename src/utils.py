import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, labels, output_prefix="model"):
    """
    Prints classification report, saves metrics to JSON,
    and creates a confusion matrix plot.

    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
        labels (list): List of all class labels in sorted order
        output_prefix (str): Used in filenames for saving reports
    """
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))

    # Save metrics to JSON
    with open(f"reports/{output_prefix}_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(f"Confusion Matrix – {output_prefix}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"reports/cm_{output_prefix}.png")

    print(f"✅ Evaluation saved to reports/cm_{output_prefix}.png and reports/{output_prefix}_metrics.json")
