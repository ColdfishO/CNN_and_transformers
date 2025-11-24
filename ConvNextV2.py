import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from timm import create_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import psutil

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

device = torch.device('cuda')
model = create_model("convnextv2_tiny", pretrained=False, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 1
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
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch + 1} finished. Loss: {running_loss / len(trainloader):.4f}, Time: {epoch_time:.1f}s")
    print(f"Used RAM: {ram_epoch:.2f} GB, GPU: {gpu_mem_epoch:.2f} GB")

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in tqdm(testloader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
cm = confusion_matrix(all_labels, all_preds)
print(f"Test Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
print("Confusion Matrix:\n", cm)