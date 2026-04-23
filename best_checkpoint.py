import torch
import os

best_score = 0
best_ckpt_path = None
best_acc_coarse = 0
best_acc_fine = 0

for epoch in range(1, 201):
    path = f"checkpoints/resnet100_checkpoints/checkpoint_epoch{epoch}.pth"

    if not os.path.exists(path):
        continue

    ckpt = torch.load(path)

    score = ckpt['val_acc_coarse'] + ckpt['val_acc_fine']

    if score > best_score:
        best_score = score
        best_acc_coarse = ckpt['val_acc_coarse']
        best_acc_fine = ckpt['val_acc_fine']
        best_ckpt_path = path

print("Best checkpoint:", best_ckpt_path,
      "val_acc_coarse:", best_acc_coarse,
      "val_acc_fine:", best_acc_fine)