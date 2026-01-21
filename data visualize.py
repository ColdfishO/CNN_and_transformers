import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Settings
# -----------------------------
ckpt_dirs = {
    "convnext": "checkpoints/convnext_checkpoints",
    "swint": "checkpoints/swint_checkpoints",
}

num_epochs = 200
output_dir = "./data_visualisation/models_compare"
os.makedirs(output_dir, exist_ok=True)

model_colors = {
    "convnext": "blue",
    "swint": "red",
}

# -----------------------------
# Metrics
# -----------------------------
metric_keys = [
    "epoch",
    "train_loss",
    "val_acc",
    "val_f1",
    "val_precision",
    "val_recall",
    "epoch_train_time",
    "epoch_avg_batch_inference_time",
    "ram_usage",
    "val_conf_matrix"
]

metric_labels = {
    "train_loss": "Train Loss",
    "val_acc": "Accuracy",
    "val_f1": "F1 Score",
    "val_precision": "Precision",
    "val_recall": "Recall",
    "epoch_train_time": "Epoch Training Time (s)",
    "epoch_avg_batch_inference_time": "Avg Batch Inference Time (s)",
    "ram_usage": "RAM Usage (GB)",
}

# -----------------------------
# Load checkpoints
# -----------------------------
def load_checkpoints(ckpt_dir):
    data = []

    for epoch in range(1, num_epochs + 1):
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch}.pth")
        if not os.path.exists(ckpt_path):
            continue

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        row = [checkpoint.get(key, None) for key in metric_keys]
        data.append(row)

    df = pd.DataFrame(data, columns=metric_keys)
    df["epoch"] = df["epoch"] + 1
    return df

# -----------------------------
# Create DataFrames + Excel
# -----------------------------
dfs = {}

for model, ckpt_dir in ckpt_dirs.items():
    df = load_checkpoints(ckpt_dir)
    dfs[model] = df

    excel_path = os.path.join(output_dir, f"{model}_metrics.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Excel saved for {model}: {excel_path}")

# -----------------------------
# Plot comparison charts (FIXED X-AXIS)
# -----------------------------
def plot_metric_compare(dfs, metric_key, y_label=None, title=None):
    plt.figure(figsize=(12, 6))

    for model, df in dfs.items():
        plt.plot(
            df["epoch"],
            df[metric_key],
            marker="o",
            linestyle="-",
            color=model_colors[model],
            label=model.upper()
        )

    plt.xlabel("Epoch")
    plt.ylabel(y_label or metric_key.replace("_", " ").title())

    # ✅ FIX: clean x-axis every 10 epochs
    plt.xticks(range(0, num_epochs + 1, 10))

    plt.grid(True)
    plt.legend()
    plt.title(title or f"{y_label} vs Epoch")
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{metric_key}_compare.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Chart saved: {save_path}")

# -----------------------------
# Generate charts
# -----------------------------
for key in metric_keys:
    if key in ["epoch", "val_conf_matrix"]:
        continue

    y_label = metric_labels.get(key, key)
    plot_metric_compare(
        dfs,
        metric_key=key,
        y_label=y_label,
        title=f"{y_label} over Epochs (ConvNeXt vs SwinT)"
    )

# -----------------------------
# Confusion matrices (EVERY 20 EPOCHS)
# -----------------------------
cm_dir = os.path.join(output_dir, "confusion_matrices")
os.makedirs(cm_dir, exist_ok=True)

for model, df in dfs.items():
    model_cm_dir = os.path.join(cm_dir, model)
    os.makedirs(model_cm_dir, exist_ok=True)

    subset_epochs = df["epoch"][(df["epoch"] == 1) | (df["epoch"] % 20 == 0)]

    for epoch in subset_epochs:
        row = df[df["epoch"] == epoch].iloc[0]
        cm = row.get("val_conf_matrix", None)

        if cm is None:
            continue

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{model.upper()} Confusion Matrix - Epoch {epoch}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        save_path = os.path.join(
            model_cm_dir, f"confusion_matrix_epoch_{epoch}.png"
        )
        plt.savefig(save_path)
        plt.close()

        print(f"Confusion matrix saved: {save_path}")
