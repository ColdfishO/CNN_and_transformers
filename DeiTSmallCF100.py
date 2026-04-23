import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from timm.models.vision_transformer import VisionTransformer
from ResNet34CF100 import HierarchicalResNet
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import time
import psutil
import os

coarse_labels_map = [
    4,1,14,8,0,6,7,7,18,3,3,14,9,18,7,11,3,9,7,11,
    6,11,5,10,7,6,13,15,3,15,0,11,1,10,12,14,16,9,11,5,
    5,19,8,8,15,13,14,17,18,10,16,4,17,4,2,0,17,4,18,17,
    10,3,2,12,12,16,12,1,9,19,2,10,0,1,16,12,9,13,15,13,
    16,19,2,4,6,19,5,5,8,19,18,1,2,15,6,0,17,8,14,13
]

coarse_to_fine = {i: [] for i in range(20)}
for fine_idx, coarse_idx in enumerate(coarse_labels_map):
    coarse_to_fine[coarse_idx].append(fine_idx)

mask_matrix = torch.zeros(20, 100, dtype=torch.bool)
for coarse, fine_list in coarse_to_fine.items():
    mask_matrix[coarse, fine_list] = True


class CIFAR100Hierarchy(torch.utils.data.Dataset):
    def __init__(self, train, transform):
        self.dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=train,
            download=True
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, fine_label = self.dataset[idx]
        coarse_label = coarse_labels_map[fine_label]
        if self.transform:
            img = self.transform(img)
        return img, fine_label, coarse_label


train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5071,0.4865,0.4409),
                (0.2673,0.2564,0.2761))
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5071,0.4865,0.4409),
                (0.2673,0.2564,0.2761))
])


trainset = CIFAR100Hierarchy(True, train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = CIFAR100Hierarchy(False, test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

device = torch.device("cuda")
mask_matrix = mask_matrix.to(device)

teacher = HierarchicalResNet().to(device)
teacher = teacher.to(device)

ckpt = torch.load("checkpoints/resnet100_checkpoints/checkpoint_epoch182.pth")
teacher.load_state_dict(ckpt["model_state_dict"])

teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False


class DeiTSmallHierarchical(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = VisionTransformer(
            img_size=32,
            patch_size=4,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            num_classes=0
        )

        dim = self.backbone.embed_dim

        self.dist_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.dist_token, std=0.02)

        num_tokens = self.backbone.patch_embed.num_patches + 2
        self.backbone.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.backbone.pos_embed, std=0.02)

        self.head_coarse = nn.Linear(dim, 20)
        self.head_fine = nn.Linear(dim, 100)

        self.dist_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 100)
        )

    def forward_features(self, x):
        x = self.backbone.patch_embed(x)

        B = x.shape[0]

        cls = self.backbone.cls_token.expand(B, -1, -1)
        dist = self.dist_token.expand(B, -1, -1)

        x = torch.cat([cls, dist, x], dim=1)

        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        for blk in self.backbone.blocks:
            x = blk(x)

        x = self.backbone.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        cls_tok = x[:, 0]
        dist_tok = x[:, 1]

        coarse_logits = self.head_coarse(cls_tok)
        fine_logits = self.head_fine(cls_tok)
        dist_logits = self.dist_proj(dist_tok)

        return coarse_logits, fine_logits, dist_logits


model = DeiTSmallHierarchical().to(device)

criterion = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction="batchmean")

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)

alpha = 0.5
temperature = 3.0

ckpt_dir = "checkpoints/deitSmall100_checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

total_params = sum(p.numel() for p in model.parameters())
print("Total params:", total_params)

num_epochs = 200

for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0
    start_time = time.time()

    for inputs, fine_labels, coarse_labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, fine_labels, coarse_labels = inputs.to(device), fine_labels.to(device), coarse_labels.to(device)
        optimizer.zero_grad()
        coarse_logits, fine_logits, dist_logits = model(inputs)
        loss_coarse = criterion(coarse_logits, coarse_labels)
        # Mask fine logits based on ground-truth coarse
        mask = mask_matrix[coarse_labels]
        fine_logits_masked = fine_logits.masked_fill(~mask, float('-inf'))
        loss_fine = criterion(fine_logits_masked, fine_labels)

        with torch.no_grad():
            _, teacher_fine_logits = teacher(inputs)

        loss_kd = kl_loss(
            torch.nn.functional.log_softmax(dist_logits / temperature, dim=1),
            torch.nn.functional.softmax(teacher_fine_logits / temperature, dim=1)
        )

        loss = loss_coarse + 0.5 * loss_fine + alpha * loss_kd

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
    ram_epoch = psutil.virtual_memory().used / (1024 ** 3)
    epoch_train_time = time.time() - start_time
    train_loss = round(running_loss / len(trainloader), 4)
    print(f"Epoch {epoch + 1} finished. Loss: {train_loss}, Time: {epoch_train_time:.1f}s")

    model.eval()
    all_fine_preds = []
    all_fine_labels = []
    all_coarse_preds = []
    all_coarse_labels = []

    total_inference_time = 0.0

    with torch.no_grad():
        for inputs, fine_labels, coarse_labels in testloader:
            inputs = inputs.to(device)
            fine_labels = fine_labels.to(device)
            coarse_labels = coarse_labels.to(device)

            torch.cuda.synchronize()
            start_time = time.time()

            coarse_logits, fine_logits, _ = model(inputs)

            torch.cuda.synchronize()
            end_time = time.time()

            batch_inference_time = end_time - start_time
            total_inference_time += batch_inference_time

            _, coarse_preds = torch.max(coarse_logits, 1)

            all_coarse_preds.extend(coarse_preds.cpu().numpy())
            all_coarse_labels.extend(coarse_labels.cpu().numpy())

            mask = mask_matrix[coarse_preds]
            fine_logits_masked = fine_logits.masked_fill(~mask, float('-inf'))

            _, fine_preds = torch.max(fine_logits_masked, 1)

            all_fine_preds.extend(fine_preds.cpu().numpy())
            all_fine_labels.extend(fine_labels.cpu().numpy())

    avg_time_per_batch = total_inference_time / len(testloader)

    # Metrics
    acc_fine = accuracy_score(all_fine_labels, all_fine_preds)
    f1_fine = f1_score(all_fine_labels, all_fine_preds, average='macro')
    precision_fine = precision_score(all_fine_labels, all_fine_preds, average='macro')
    recall_fine = recall_score(all_fine_labels, all_fine_preds, average='macro')
    cm_fine = confusion_matrix(all_fine_labels, all_fine_preds)

    acc_coarse = accuracy_score(all_coarse_labels, all_coarse_preds)
    f1_coarse = f1_score(all_coarse_labels, all_coarse_preds, average='macro')
    precision_coarse = precision_score(all_coarse_labels, all_coarse_preds, average='macro')
    recall_coarse = recall_score(all_coarse_labels, all_coarse_preds, average='macro')
    cm_coarse = confusion_matrix(all_coarse_labels, all_coarse_preds)

    print(
        f"Test Fine Acc: {acc_fine:.4f}, F1: {f1_fine:.4f}, Precision: {precision_fine:.4f}, Recall: {recall_fine:.4f}")
    print("Fine Confusion Matrix:\n", cm_fine)
    print(
        f"Test Coarse Acc: {acc_coarse:.4f}, F1: {f1_coarse:.4f}, Precision: {precision_coarse:.4f}, Recall: {recall_coarse:.4f}")
    print("Coarse Confusion Matrix:\n", cm_coarse)
    print(f"Used RAM: {ram_epoch:.2f} GB, Avg inference time per batch: {avg_time_per_batch:.6f} sec")

    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch + 1}.pth")
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