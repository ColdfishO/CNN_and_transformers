import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Settings
# -----------------------------
ckpt_dir = "checkpoints/swint_checkpoints"
num_epochs = 100
output_dir = "./data_visualisation/modelswint"
os.makedirs(output_dir, exist_ok=True)

# List of metrics to extract
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
    "gpu_usage",
    "val_conf_matrix"
]

# Optional: human-readable labels for chart y-axis
metric_labels = {
    "train_loss": "Train Loss",
    "val_acc": "Accuracy",
    "val_f1": "F1 Score",
    "val_precision": "Precision",
    "val_recall": "Recall",
    "epoch_train_time": "Epoch Training Time (s)",
    "epoch_avg_batch_inference_time": "Avg Batch Inference Time (s)",
    "ram_usage": "RAM Usage (GB)",
    "gpu_usage": "GPU Usage (GB)"
}

# -----------------------------
# Load checkpoints
# -----------------------------
data = []

for epoch in range(1, num_epochs + 1):
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch}.pth")
    if not os.path.exists(ckpt_path):
        print(f"Warning: checkpoint {ckpt_path} not found.")
        continue

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    row = [checkpoint.get(key, None) for key in metric_keys]
    data.append(row)

# -----------------------------
# Create DataFrame
# -----------------------------
df = pd.DataFrame(data, columns=metric_keys)
df['epoch'] = df['epoch'] + 1
print(df.head())

# -----------------------------
# Save DataFrame as Excel
# -----------------------------
excel_path = os.path.join(output_dir, "metrics_table.xlsx")
df.to_excel(excel_path, index=False)
print(f"Metrics table saved to {excel_path}")

# -----------------------------
# Plotting function
# -----------------------------
def plot_metric(df, metric_key, y_label=None, title=None, tick_every=5):
    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df[metric_key], marker='o', linestyle='-', color='red')
    plt.xlabel("Epoch")
    plt.ylabel(y_label or metric_key.replace("_", " ").title())
    plt.xticks(df['epoch'][::tick_every])
    plt.grid(True)
    plt.title(title or f"{y_label or metric_key.replace('_',' ').title()} vs Epoch")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{metric_key}_chart.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Chart saved to {save_path}")

# -----------------------------
# Generate chart for each metric
# -----------------------------
for key in metric_keys:
    if key in ["epoch", "val_conf_matrix"]:
        continue  # skip epoch itself and confusion matrices
    y_label = metric_labels.get(key, key)
    plot_metric(df, metric_key=key, y_label=y_label, title=f"{y_label} over Epochs", tick_every=5)

# -----------------------------
# Confusion matrix heatmaps every 10 epochs
# -----------------------------
cm_dir = os.path.join(output_dir, "confusion_matrices")
os.makedirs(cm_dir, exist_ok=True)

# Select every 10th epoch
subset_epochs = df['epoch'][(df['epoch'] == 1) | (df['epoch'] % 10 == 0)]

for epoch in subset_epochs:
    row = df[df['epoch'] == epoch].iloc[0]
    cm = row.get('val_conf_matrix', None)
    if cm is None:
        continue

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix - Epoch {epoch}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    save_path = os.path.join(cm_dir, f"confusion_matrix_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix heatmap saved for epoch {epoch}")
