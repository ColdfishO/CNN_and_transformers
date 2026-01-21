import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from timm import create_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import time
import psutil
import os
import pickle

device = torch.device('cuda')


# Minimal augmentation to match your CIFAR-10 setup
transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5071,0.4865,0.4409), (0.2673,0.2564,0.2761))
])

# Custom dataset to return fine + coarse labels
class CIFAR100WithSuperclasses(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__(root=root, train=train, transform=transform, download=download)
        # Load coarse labels
        base_folder = self.base_folder
        file_list = self.train_list if train else self.test_list
        path = os.path.join(self.root, base_folder, file_list[0][0])
        with open(path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.coarse_labels = entry['coarse_labels']

    def __getitem__(self, index):
        img, fine_label = super().__getitem__(index)
        coarse_label = self.coarse_labels[index]
        return img, fine_label, coarse_label

# Datasets and loaders
trainset = CIFAR100WithSuperclasses(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = CIFAR100WithSuperclasses(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)


class ConvNeXtV2Hierarchical(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = create_model("convnextv2_tiny", pretrained=False, num_classes=0)
        self.fine_head = nn.Linear(768, 100)   # fine labels
        self.coarse_head = nn.Linear(768, 20)  # superclasses

    def forward(self, x):
        features = self.backbone.forward_features(x)  # [B, 768, H, W]
        features = features.mean(dim=[2, 3])  # [B, 768]
        fine_out = self.fine_head(features)
        coarse_out = self.coarse_head(features)
        return fine_out, coarse_out


model = ConvNeXtV2Hierarchical().to(device)


criterion_fine = nn.CrossEntropyLoss()
criterion_coarse = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-2)


num_epochs = 200
ckpt_dir = "checkpoints/convnext100_checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)
print(f"Initial RAM memory: {psutil.virtual_memory().used / (1024**3):.2f} GB")


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    correct_fine = 0
    correct_coarse = 0
    total = 0
    for inputs, fine_labels, coarse_labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, fine_labels, coarse_labels = inputs.to(device), fine_labels.to(device), coarse_labels.to(device)

        optimizer.zero_grad()
        outputs_fine, outputs_coarse = model(inputs)

        # Multi-task loss
        loss_fine = criterion_fine(outputs_fine, fine_labels)
        loss_coarse = criterion_coarse(outputs_coarse, coarse_labels)
        loss = loss_fine + loss_coarse
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    ram_epoch = psutil.virtual_memory().used / (1024 ** 3)
    epoch_train_time = time.time() - start_time
    train_loss = round(running_loss / len(trainloader), 4)
    print(f"Epoch {epoch+1} finished. Loss: {train_loss}, Time: {epoch_train_time:.1f}s")

    model.eval()
    all_preds_fine = []
    all_labels_fine = []
    all_preds_coarse = []
    all_labels_coarse = []
    total_inference_time = 0.0

    with torch.no_grad():
        for inputs, fine_labels, coarse_labels in testloader:
            inputs, fine_labels, coarse_labels = inputs.to(device), fine_labels.to(device), coarse_labels.to(device)

            torch.cuda.synchronize()
            start_time = time.time()
            outputs_fine, outputs_coarse = model(inputs)
            torch.cuda.synchronize()
            end_time = time.time()
            batch_inference_time = end_time - start_time
            total_inference_time += batch_inference_time

            _, preds_fine = torch.max(outputs_fine, 1)
            _, preds_coarse = torch.max(outputs_coarse, 1)
            all_preds_fine.extend(preds_fine.cpu().numpy())
            all_labels_fine.extend(fine_labels.cpu().numpy())
            all_preds_coarse.extend(preds_coarse.cpu().numpy())
            all_labels_coarse.extend(coarse_labels.cpu().numpy())

    avg_time_per_batch = total_inference_time / len(testloader)

    # Metrics
    acc_fine = accuracy_score(all_labels_fine, all_preds_fine)
    f1_fine = f1_score(all_labels_fine, all_preds_fine, average='macro')
    precision_fine = precision_score(all_labels_fine, all_preds_fine, average='macro')
    recall_fine = recall_score(all_labels_fine, all_preds_fine, average='macro')
    cm_fine = confusion_matrix(all_labels_fine, all_preds_fine)

    acc_coarse = accuracy_score(all_labels_coarse, all_preds_coarse)
    f1_coarse = f1_score(all_labels_coarse, all_preds_coarse, average='macro')
    precision_coarse = precision_score(all_labels_coarse, all_preds_coarse, average='macro')
    recall_coarse = recall_score(all_labels_coarse, all_preds_coarse, average='macro')
    cm_coarse = confusion_matrix(all_labels_coarse, all_preds_coarse)

    print(f"Test Fine Acc: {acc_fine:.4f}, F1: {f1_fine:.4f}, Precision: {precision_fine:.4f}, Recall: {recall_fine:.4f}")
    print("Fine Confusion Matrix:\n", cm_fine)
    print(f"Test Coarse Acc: {acc_coarse:.4f}, F1: {f1_coarse:.4f}, Precision: {precision_coarse:.4f}, Recall: {recall_coarse:.4f}")
    print("Coarse Confusion Matrix:\n", cm_coarse)
    print(f"Used RAM: {ram_epoch:.2f} GB, Avg inference time per batch: {avg_time_per_batch:.6f} sec")


    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch+1}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_acc_fine': acc_fine,
        'val_f1_fine': f1_fine,
        'val_precision_fine': precision_fine,
        'val_recall_fine': recall_fine,
        'val_conf_matrix_fine': cm_fine,
        'val_acc_coarse': acc_coarse,
        'val_f1_coarse': f1_coarse,
        'val_precision_coarse': precision_coarse,
        'val_recall_coarse': recall_coarse,
        'val_conf_matrix_coarse': cm_coarse,
        'epoch_train_time': epoch_train_time,
        'epoch_avg_batch_inference_time': avg_time_per_batch,
        'ram_usage': ram_epoch
    }, ckpt_path)
