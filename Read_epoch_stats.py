import torch
import os

# -----------------------------
# Path to the checkpoint
# -----------------------------
ckpt_dir = "checkpoints/convnext_checkpoints"
epoch_number = 1  # change to the epoch you want to load
ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch_number}.pth")

# -----------------------------
# Load checkpoint
# -----------------------------
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))  # or 'cuda' if desired

# -----------------------------
# Print metrics
# -----------------------------
print(f"Epoch: {checkpoint['epoch'] + 1}")  # add 1 if you stored zero-based
print(f"Train Loss: {checkpoint['train_loss']:.4f}")
print(f"Validation Accuracy: {checkpoint['val_acc']:.4f}")
print(f"F1 Score: {checkpoint['val_f1']:.4f}")
print(f"Precision: {checkpoint['val_precision']:.4f}")
print(f"Recall: {checkpoint['val_recall']:.4f}")
print(f"Confusion Matrix:\n{checkpoint['val_conf_matrix']}")
print(f"Epoch Time (s): {checkpoint['epoch_time']:.2f}")
print(f"RAM Usage (GB): {checkpoint['ram_usage']:.2f}")
print(f"GPU Usage (GB): {checkpoint['gpu_usage']:.2f}")
