import torch

best_acc = 0
best_ckpt_path = None
for epoch in range(1, 201):
    path = f"checkpoints/resnet_checkpoints/checkpoint_epoch{epoch}.pth"
    ckpt = torch.load(path)
    if ckpt['val_acc'] > best_acc:
        best_acc = ckpt['val_acc']
        best_ckpt_path = path

print("Best checkpoint:", best_ckpt_path, "with val_acc:", best_acc)