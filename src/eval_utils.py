def evaluate_model(y_true, y_pred, labels, output_prefix):
    from sklearn.metrics import classification_report, confusion_matrix
    import json, matplotlib.pyplot as plt, seaborn as sns

    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))

    with open(f"reports/{output_prefix}_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix – {output_prefix}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"reports/cm_{output_prefix}.png")