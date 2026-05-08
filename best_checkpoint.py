import torch
import os

best_score = -1
best_ckpt_path = None

for epoch in range(1, 201):

    path = f"checkpoints/resnetSTL_checkpoints/checkpoint_epoch{epoch}.pth"

    if not os.path.exists(path):
        continue

    ckpt = torch.load(path)

    acc = ckpt['val_acc']
    f1 = ckpt['val_f1']

    # balanced metric
    score = 0.7 * acc + 0.3 * f1

    if score > best_score:
        best_score = score
        best_ckpt_path = path

        best_acc = acc
        best_f1 = f1
        best_precision = ckpt['val_precision']
        best_recall = ckpt['val_recall']
        best_epoch = ckpt['epoch'] + 1

print("\n=== BEST CHECKPOINT ===")
print("Checkpoint:", best_ckpt_path)
print("Epoch:", best_epoch)
print(f"Accuracy : {best_acc:.4f}")
print(f"F1 Score : {best_f1:.4f}")
print(f"Precision: {best_precision:.4f}")
print(f"Recall   : {best_recall:.4f}")
print(f"Combined Score: {best_score:.4f}")