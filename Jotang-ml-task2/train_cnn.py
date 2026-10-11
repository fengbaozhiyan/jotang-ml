import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# ===================== 工具函数：过滤损坏图片 =====================
def filter_broken_images(root_dir):
    valid_imgs = []
    valid_labels = []
    class_map = [(0, "Cat"), (1, "Dog")]
    for label, folder_name in class_map:
        folder_path = os.path.join(root_dir, folder_name)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".jpg"):
                full_path = os.path.join(folder_path, filename)
                try:
                    with Image.open(full_path) as im:
                        im.verify()
                    valid_imgs.append(full_path)
                    valid_labels.append(label)
                except Exception:
                    print(f"跳过损坏图片: {full_path}")
    print(f"有效图片总数：{len(valid_imgs)}")
    return valid_imgs, valid_labels

# ===================== 数据集类 =====================
class CatDogDataset(Dataset):
    def __init__(self, img_path_list, label_list, transform=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transform = transform
    def __len__(self):
        return len(self.img_path_list)
    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        label = self.label_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ===================== 数据预处理 =====================
transform_base = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
transform_aug = transforms.Compose([
    transforms.Resize((140, 140)),
    transforms.RandomCrop((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

img_dir = "PetImages"
valid_imgs, valid_labels = filter_broken_images(img_dir)
full_ds_base = CatDogDataset(valid_imgs, valid_labels, transform=transform_base)
full_ds_aug = CatDogDataset(valid_imgs, valid_labels, transform=transform_aug)

train_ds_base, val_ds_base = random_split(full_ds_base, [20000, len(full_ds_base)-20000])
train_ds_aug, val_ds_aug = random_split(full_ds_aug, [20000, len(full_ds_aug)-20000])

BATCH_SIZE = 16
train_loader_base = DataLoader(train_ds_base, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader_base = DataLoader(val_ds_base, batch_size=BATCH_SIZE, shuffle=False)
train_loader_aug = DataLoader(train_ds_aug, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader_aug = DataLoader(val_ds_aug, batch_size=BATCH_SIZE, shuffle=False)

# ===================== CNN模型定义 =====================
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CatDogCNN()
# 测试张量维度
test_tensor = torch.randn((1, 3, 128, 128))
out = model(test_tensor)

# ===================== 第一层卷积特征图可视化 =====================
feature_map = []
def hook_fn(module, input, output):
    feature_map.append(output)
handle = model.conv1.register_forward_hook(hook_fn)
img_sample, _ = val_loader_base.dataset[0]
model.eval()
with torch.no_grad():
    model(img_sample.unsqueeze(0))
fm = feature_map[0].squeeze(0).cpu().numpy()
plt.figure(figsize=(10, 6))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(fm[i], cmap="gray")
    plt.title(f"ch{i}")
    plt.axis("off")
os.makedirs("output", exist_ok=True)
plt.savefig("output/feature_map.jpg")
plt.show()
handle.remove()

# ===================== CPU训练循环 =====================
device = torch.device("cpu")
model = CatDogCNN().to(device)
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
epoch_num = 10
train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []

for epoch in range(epoch_num):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader_aug:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        pred = model(imgs)
        loss = criterion(pred, labels)
        loss.backward()
        opt.step()
        train_loss += loss.item()
        _, predicted = torch.max(pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_acc = correct / total
    # ========= 计算平均训练loss =========
    avg_train_loss = train_loss / len(train_loader_aug)

    # 验证集评估
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for imgs, labels in val_loader_aug:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs)
            loss = criterion(pred, labels)
            val_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    val_acc = correct_val / total_val
    # ========= 计算平均验证loss =========
    avg_val_loss = val_loss / len(val_loader_aug)

    train_loss_list.append(avg_train_loss)
    val_loss_list.append(avg_val_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    print(f"Epoch{epoch}, train_loss:{avg_train_loss:.3f}, train_acc:{train_acc:.3f}, val_loss:{avg_val_loss:.3f}, val_acc:{val_acc:.3f}")

# 保存模型
torch.save(model.state_dict(), "catdog_best.pth")

# ======================== 新增：绘制Loss & Accuracy曲线 ========================
plt.figure(figsize=(12, 5))
epochs = list(range(epoch_num))

# 子图1 Loss曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_list, 'o-', label="Train Loss")
plt.plot(epochs, val_loss_list, 's-', label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2 Accuracy曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc_list, 'o-', label="Train Acc")
plt.plot(epochs, val_acc_list, 's-', label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/loss_acc_curve.jpg")
plt.show()

# ===================== 查找并可视化预测错误样本 =====================
model.eval()
err_img_list = []
err_label_true = []
err_label_pred = []
with torch.no_grad():
    for imgs, labels in val_loader_base:
        imgs = imgs.to(device)
        pred = model(imgs)
        _, pred_idx = torch.max(pred, dim=1)
        for i in range(len(labels)):
            if pred_idx[i] != labels[i]:
                err_img_list.append(imgs[i].cpu())
                err_label_true.append(labels[i].item())
                err_label_pred.append(pred_idx[i].item())
            if len(err_img_list) >= 5:
                break
        if len(err_img_list) >= 5:
            break
plt.figure(figsize=(8, 6))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = err_img_list[i].permute(1,2,0).numpy()
    plt.imshow(img)
    plt.title(f"true:{err_label_true[i]}, pred:{err_label_pred[i]}")
    plt.axis("off")
plt.savefig("output/error_sample.jpg")
plt.show()

