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
    "googlenet": "checkpoints/googlenet100_checkpoints",
    "resnet": "checkpoints/resnet100_checkpoints",
    "swint": "checkpoints/swint100_checkpoints",
    "vitSmall": "checkpoints/vitSmall100_checkpoints",
    "deitSmall": "checkpoints/deitSmall100_checkpoints",
}

num_epochs = 200
output_dir = "./data_visualisation/models_compare_cifar100"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Colors
# -----------------------------
model_colors = {
    "convnext": "blue",
    "googlenet": "green",
    "resnet": "purple",
    "swint": "red",
    "vitSmall": "orange",
    "deitSmall": "gold",
}

# -----------------------------
# Metrics
# -----------------------------
metric_keys = [
    "epoch",
    "train_loss",

    # Fine metrics
    "val_acc_fine",
    "val_f1_fine",
    "val_precision_fine",
    "val_recall_fine",
    "val_conf_matrix_fine",

    # Coarse metrics
    "val_acc_coarse",
    "val_f1_coarse",
    "val_precision_coarse",
    "val_recall_coarse",
    "val_conf_matrix_coarse",

    # System metrics
    "epoch_train_time",
    "epoch_avg_batch_inference_time",
    "ram_usage",
]

metric_labels = {
    "train_loss": "Train Loss",

    "val_acc_fine": "Fine Accuracy",
    "val_f1_fine": "Fine F1 Score",
    "val_precision_fine": "Fine Precision",
    "val_recall_fine": "Fine Recall",

    "val_acc_coarse": "Coarse Accuracy",
    "val_f1_coarse": "Coarse F1 Score",
    "val_precision_coarse": "Coarse Precision",
    "val_recall_coarse": "Coarse Recall",

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

        ckpt_path = os.path.join(
            ckpt_dir,
            f"checkpoint_epoch{epoch}.pth"
        )

        if not os.path.exists(ckpt_path):
            continue

        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu"
        )

        # remove heavy objects
        checkpoint.pop("model_state_dict", None)
        checkpoint.pop("optimizer_state_dict", None)

        row = [
            checkpoint.get(key, None)
            for key in metric_keys
        ]

        data.append(row)

    df = pd.DataFrame(
        data,
        columns=metric_keys
    )

    df["epoch"] = df["epoch"] + 1

    return df

# -----------------------------
# Create DataFrames + Excel
# -----------------------------
dfs = {}

for model, ckpt_dir in ckpt_dirs.items():

    df = load_checkpoints(ckpt_dir)

    dfs[model] = df

    excel_path = os.path.join(
        output_dir,
        f"{model}_metrics.xlsx"
    )

    df.to_excel(excel_path, index=False)

    print(f"Excel saved for {model}: {excel_path}")

# -----------------------------
# Plot comparison charts
# -----------------------------
def plot_metric_compare(
    dfs,
    metric_key,
    y_label=None,
    title=None
):
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

    plt.ylabel(
        y_label or metric_key.replace("_", " ").title()
    )

    # fixed x-axis
    plt.xticks(range(0, num_epochs + 1, 10))

    plt.grid(True)

    plt.legend()

    plt.title(
        title or f"{y_label} vs Epoch"
    )

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        f"{metric_key}_compare.png"
    )

    plt.savefig(save_path)

    plt.close()

    print(f"Chart saved: {save_path}")

# -----------------------------
# Generate charts
# -----------------------------
for key in metric_keys:

    if key in [
        "epoch",
        "val_conf_matrix_fine",
        "val_conf_matrix_coarse"
    ]:
        continue

    y_label = metric_labels.get(key, key)

    plot_metric_compare(
        dfs,
        metric_key=key,
        y_label=y_label,
        title=f"{y_label} over Epochs (Model Comparison)"
    )

# -----------------------------
# Confusion matrices
# (EVERY 20 EPOCHS)
# -----------------------------
cm_dir = os.path.join(
    output_dir,
    "confusion_matrices"
)

os.makedirs(cm_dir, exist_ok=True)

for model, df in dfs.items():

    model_cm_dir = os.path.join(
        cm_dir,
        model
    )

    os.makedirs(model_cm_dir, exist_ok=True)

    subset_epochs = df["epoch"][
        (df["epoch"] == 1)
        | (df["epoch"] % 20 == 0)
    ]

    for epoch in subset_epochs:

        row = df[df["epoch"] == epoch].iloc[0]

        # -----------------------------
        # Fine confusion matrix
        # -----------------------------
        fine_cm = row.get(
            "val_conf_matrix_fine",
            None
        )

        if fine_cm is not None:

            fine_dir = os.path.join(
                model_cm_dir,
                "fine"
            )

            os.makedirs(fine_dir, exist_ok=True)

            plt.figure(figsize=(12, 10))

            sns.heatmap(
                fine_cm,
                annot=False,
                cmap="Blues",
                cbar=True
            )

            plt.title(
                f"{model.upper()} Fine Confusion Matrix - Epoch {epoch}"
            )

            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            plt.tight_layout()

            save_path = os.path.join(
                fine_dir,
                f"confusion_matrix_epoch_{epoch}.png"
            )

            plt.savefig(save_path)

            plt.close()

            print(f"Fine confusion matrix saved: {save_path}")

        # -----------------------------
        # Coarse confusion matrix
        # -----------------------------
        coarse_cm = row.get(
            "val_conf_matrix_coarse",
            None
        )

        if coarse_cm is not None:

            coarse_dir = os.path.join(
                model_cm_dir,
                "coarse"
            )

            os.makedirs(coarse_dir, exist_ok=True)

            plt.figure(figsize=(8, 6))

            sns.heatmap(
                coarse_cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False
            )

            plt.title(
                f"{model.upper()} Coarse Confusion Matrix - Epoch {epoch}"
            )

            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            plt.tight_layout()

            save_path = os.path.join(
                coarse_dir,
                f"confusion_matrix_epoch_{epoch}.png"
            )

            plt.savefig(save_path)

            plt.close()

            print(f"Coarse confusion matrix saved: {save_path}")