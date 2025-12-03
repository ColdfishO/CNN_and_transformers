import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from timm.models.swin_transformer import SwinTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import os

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

device = torch.device('cuda')
model = SwinTransformer(img_size=32, patch_size=2, window_size=2, in_chans=3, num_classes=10, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.05)

total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)

ckpt_dir = "checkpoints/swint_checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

num_epochs = 100
print(f"Initial RAM memory: {psutil.virtual_memory().used / (1024**3):.2f} GB")
print(f"Initial GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    ram_epoch = psutil.virtual_memory().used / (1024 ** 3)
    gpu_mem_epoch = torch.cuda.memory_allocated() / (1024 ** 3)
    epoch_train_time = time.time() - start_time
    train_loss = round(running_loss / len(trainloader), 4)
    print(f"Epoch {epoch + 1} finished. Loss: {train_loss}, Time: {epoch_train_time:.1f}s")
    model.eval()
    all_preds = []
    all_labels = []
    total_inference_time = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            torch.cuda.synchronize()
            start_time = time.time()
            outputs = model(inputs)
            torch.cuda.synchronize()
            end_time = time.time()
            batch_inference_time = end_time - start_time
            total_inference_time += batch_inference_time
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_time_per_batch = total_inference_time / len(testloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print("Confusion Matrix:\n", cm)
    print(f"Used RAM: {ram_epoch:.2f} GB, GPU: {gpu_mem_epoch:.2f} GB")
    print(f"Inference time: {avg_time_per_batch:.6f} sec")
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch+1}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_acc': acc,
        'val_f1': f1,
        'val_precision': precision,
        'val_recall': recall,
        'val_conf_matrix': cm,
        'epoch_train_time': epoch_train_time,
        'epoch_avg_batch_inference_time': avg_time_per_batch,
        'ram_usage': ram_epoch,
        'gpu_usage': gpu_mem_epoch
    }, ckpt_path)