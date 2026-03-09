import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from timm.models.vision_transformer import VisionTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import time
import psutil
import os

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(
        (0.5071, 0.4865, 0.4409),
        (0.2673, 0.2564, 0.2761)
    )
])
test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(
        (0.5071, 0.4865, 0.4409),
        (0.2673, 0.2564, 0.2761)
    )
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

device = torch.device('cuda')

# Teacher model (pretrained ResNet)
teacher = torchvision.models.resnet34(weights=None)  # Replace with pretrained weights if available
teacher.fc = nn.Linear(512, 10)  # Adjust for CIFAR-10 classes
teacher_ckpt = torch.load("checkpoints/resnet_checkpoints/checkpoint_epoch199.pth")
teacher.load_state_dict(teacher_ckpt['model_state_dict'])
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
teacher = teacher.to(device)

class DeiTSmallCIFAR(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add learnable distillation token
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.dist_token, std=.02)
        # Extra classifier head for distillation token
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        # Fix positional embedding for dist token
        num_tokens = self.patch_embed.num_patches + 2  # CLS + Distillation + patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        B = x.shape[0]
        # Original patch embedding
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)  # distillation token
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_token_final = x[:, 0]  # original CLS
        dist_token_final = x[:, 1]  # distillation token

        x_cls = self.head(cls_token_final)
        x_dist = self.head_dist(dist_token_final)

        return x_cls, x_dist  # return both for combined loss


model = DeiTSmallCIFAR(img_size=32, patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()

# KL-divergence for distillation
kl_loss = nn.KLDivLoss(reduction='batchmean')

# Alpha = weight of distillation loss
alpha = 0.5
temperature = 3.0

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)

total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)

ckpt_dir = "checkpoints/deitSmall_checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

num_epochs = 200
print(f"Initial RAM memory: {psutil.virtual_memory().used / (1024**3):.2f} GB")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits_cls, logits_dist = model(inputs)
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        loss_ce = criterion(logits_cls, labels)
        # Distillation loss (KL divergence)
        loss_kd = kl_loss(
            nn.functional.log_softmax(logits_dist / temperature, dim=1),
            nn.functional.softmax(teacher_logits / temperature, dim=1)
        )
        # Combine losses
        loss = (1 - alpha) * loss_ce + alpha * loss_kd
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    ram_epoch = psutil.virtual_memory().used / (1024 ** 3)
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
            logits_cls, _ = model(inputs)
            torch.cuda.synchronize()
            end_time = time.time()
            batch_inference_time = end_time - start_time
            total_inference_time += batch_inference_time
            _, preds = torch.max(logits_cls, 1)
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
    print(f"Used RAM: {ram_epoch:.2f} GB")
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
        'ram_usage': ram_epoch
    }, ckpt_path)