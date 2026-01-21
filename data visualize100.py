import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Settings
# -----------------------------
ckpt_dirs = {
    "convnext": "checkpoints/convnext100_checkpoints",
    "swint": "checkpoints/swint100_checkpoints",
}

num_epochs = 200
output_dir = "./data_visualisation/cifar100_compare"
os.makedirs(output_dir, exist_ok=True)

# Line colors
line_styles = {
    ("convnext", "coarse"): "blue",
    ("convnext", "fine"): "#6baed6",
    ("swint", "coarse"): "red",
    ("swint", "fine"): "#fb6a6a",
}

model_colors = {
    "convnext": "blue",
    "swint": "red",
}

# -----------------------------
# Metric keys
# -----------------------------
metric_keys = [
    "epoch",
    "train_loss",

    "val_acc_fine",
    "val_f1_fine",
    "val_precision_fine",
    "val_recall_fine",
    "val_conf_matrix_fine",

    "val_acc_coarse",
    "val_f1_coarse",
    "val_precision_coarse",
    "val_recall_coarse",
    "val_conf_matrix_coarse",

    "epoch_train_time",
    "epoch_avg_batch_inference_time",
    "ram_usage",
]

# Split metrics
metric_groups = {
    "Accuracy": ["val_acc_fine", "val_acc_coarse"],
    "F1 Score": ["val_f1_fine", "val_f1_coarse"],
    "Precision": ["val_precision_fine", "val_precision_coarse"],
    "Recall": ["val_recall_fine", "val_recall_coarse"],
}

# Non-split metrics
single_metrics = {
    "train_loss": "Train Loss",
    "epoch_train_time": "Epoch Training Time (s)",
    "epoch_avg_batch_inference_time": "Avg Batch Inference Time (s)",
    "ram_usage": "RAM Usage (GB)",
}

# -----------------------------
# Load checkpoints
# -----------------------------
def load_checkpoints(ckpt_dir):
    data = []

    for epoch in range(num_epochs):
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch+1}.pth")
        if not os.path.exists(ckpt_path):
            continue

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        row = {key: checkpoint.get(key, None) for key in metric_keys}
        data.append(row)

    df = pd.DataFrame(data)
    df["epoch"] = df["epoch"] + 1
    return df

# -----------------------------
# Load data + save Excel
# -----------------------------
dfs = {}

for model, ckpt_dir in ckpt_dirs.items():
    df = load_checkpoints(ckpt_dir)
    dfs[model] = df

    excel_path = os.path.join(output_dir, f"{model}_cifar100_metrics.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Excel saved: {excel_path}")

# -----------------------------
# Plot coarse + fine metrics
# -----------------------------
def plot_metric_group(dfs, title, fine_key, coarse_key):
    plt.figure(figsize=(12, 6))

    for model, df in dfs.items():
        plt.plot(
            df["epoch"], df[coarse_key],
            marker="o", linestyle="-",
            color=line_styles[(model, "coarse")],
            label=f"{model.upper()} Coarse"
        )

        plt.plot(
            df["epoch"], df[fine_key],
            marker="o", linestyle="-",
            color=line_styles[(model, "fine")],
            label=f"{model.upper()} Fine"
        )

    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(f"{title} over Epochs (CIFAR-100)")
    plt.xticks(range(0, num_epochs + 1, 10))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(
        output_dir, f"{title.lower().replace(' ', '_')}_compare.png"
    )
    plt.savefig(save_path)
    plt.close()

    print(f"Chart saved: {save_path}")

# -----------------------------
# Plot non-split metrics
# -----------------------------
def plot_single_metric(dfs, metric_key, y_label):
    plt.figure(figsize=(12, 6))

    for model, df in dfs.items():
        plt.plot(
            df["epoch"], df[metric_key],
            marker="o", linestyle="-",
            color=model_colors[model],
            label=model.upper()
        )

    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(f"{y_label} over Epochs (CIFAR-100)")
    plt.xticks(range(0, num_epochs + 1, 10))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{metric_key}_compare.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Chart saved: {save_path}")

# -----------------------------
# Generate plots
# -----------------------------
for title, (fine_key, coarse_key) in metric_groups.items():
    plot_metric_group(dfs, title, fine_key, coarse_key)

for metric_key, label in single_metrics.items():
    plot_single_metric(dfs, metric_key, label)

# -----------------------------
# Confusion matrices (every 20 epochs)
# -----------------------------
cm_root = os.path.join(output_dir, "confusion_matrices")
os.makedirs(cm_root, exist_ok=True)

for model, df in dfs.items():
    for level in ["fine", "coarse"]:
        cm_dir = os.path.join(cm_root, model, level)
        os.makedirs(cm_dir, exist_ok=True)

        cm_key = f"val_conf_matrix_{level}"
        subset_epochs = df["epoch"][(df["epoch"] == 1) | (df["epoch"] % 20 == 0)]

        for epoch in subset_epochs:
            row = df[df["epoch"] == epoch].iloc[0]
            cm = row.get(cm_key, None)

            if cm is None:
                continue

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
            plt.title(f"{model.upper()} {level.capitalize()} CM - Epoch {epoch}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()

            save_path = os.path.join(cm_dir, f"confusion_matrix_epoch_{epoch}.png")
            plt.savefig(save_path)
            plt.close()

            print(f"CM saved: {save_path}")
