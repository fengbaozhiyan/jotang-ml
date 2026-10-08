import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ===================== 1. 固定随机种子，保证实验可复现 =====================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# 保证卷积等确定性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===================== 2. 生成数据集 & 划分 =====================
# make_moons：2个月牙，二分类，noise控制噪声大小
X, y = make_moons(n_samples=2000, noise=0.2, random_state=SEED)

# 划分：先分出测试集，再从剩余分出训练+验证集
# 【关键：先切测试集，再切训练验证，避免数据泄露】
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=SEED, stratify=y_train_val
)

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

# 数据集职责说明
"""
训练集(train): 用来更新模型权重，拟合数据分布
验证集(val): 训练过程中看泛化能力，调超参、早停，**不能用来更新权重**
测试集(test): 全程训练阶段不可见，最终一次性评估模型真实性能；
> 禁止：训练时看测试集指标、用测试集调参，这就是数据泄露
"""

# 可视化原始数据
plt.figure(figsize=(6,5))
plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1], c="blue", alpha=0.6, label="Train class 0")
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], c="orange", alpha=0.6, label="Train class 1")
plt.scatter(X_val[y_val==0,0], X_val[y_val==0,1], c="deepskyblue", alpha=0.6, marker="s", label="Val class 0")
plt.scatter(X_val[y_val==1,0], X_val[y_val==1,1], c="darkorange", alpha=0.6, marker="s", label="Val class 1")
plt.scatter(X_test[y_test==0,0], X_test[y_test==0,1], c="navy", alpha=0.6, marker="^", label="Test class 0")
plt.scatter(X_test[y_test==1,0], X_test[y_test==1,1], c="red", alpha=0.6, marker="^", label="Test class 1")
plt.title("make_moons Dataset (Train/Val/Test split)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.tight_layout()
plt.show()

# ===================== 3. Pytorch Dataset & DataLoader =====================
class MoonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = MoonDataset(X_train, y_train)
val_ds = MoonDataset(X_val, y_val)
test_ds = MoonDataset(X_test, y_test)

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ===================== 4. 定义MLP模型 =====================
class MLP(nn.Module):
    def __init__(self, hidden_dim=64, num_hidden_layers=1, act=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = 2
        # 输入层 -> 第一个隐藏层
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act())
        # 追加额外隐藏层
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act())
        # 输出层：二分类，输出logit，后面用BCEWithLogitsLoss
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ===================== 5. 训练函数 =====================
def train_one_epoch(model, loader, loss_fn, opt, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb).squeeze()
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
        pred = (torch.sigmoid(logits) > 0.5).float()
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb).squeeze()
        loss = loss_fn(logits, yb)
        total_loss += loss.item() * xb.size(0)
        pred = (torch.sigmoid(logits) > 0.5).float()
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

# ===================== 6. 主训练流程 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# 基础模型配置
model = MLP(hidden_dim=64, num_hidden_layers=1, act=nn.ReLU).to(device)
loss_fn = nn.BCEWithLogitsLoss()
lr = 1e-3
opt = optim.Adam(model.parameters(), lr=lr)
EPOCHS = 80

train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

for epoch in range(EPOCHS):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, opt, device)
    va_loss, va_acc = eval_epoch(model, val_loader, loss_fn, device)
    train_loss_list.append(tr_loss)
    val_loss_list.append(va_loss)
    train_acc_list.append(tr_acc)
    val_acc_list.append(va_acc)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train loss:{tr_loss:.4f}, acc:{tr_acc:.4f} | Val loss:{va_loss:.4f}, acc:{va_acc:.4f}")

# 在测试集上评估【仅最后做一次！】
test_loss, test_acc = eval_epoch(model, test_loader, loss_fn, device)
print(f"\n==== Final Test ====")
print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

# 模型保存 & 加载示例
torch.save(model.state_dict(), "mlp_moon_baseline.pth")
# 加载
model_load = MLP(hidden_dim=64, num_hidden_layers=1, act=nn.ReLU).to(device)
model_load.load_state_dict(torch.load("mlp_moon_baseline.pth"))
model_load.eval()

# ===================== 7. 绘制 Loss & Accuracy 曲线 =====================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===================== 8. 绘制二维决策边界 =====================
@torch.no_grad()
def plot_decision_boundary(model, X, y, device):
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)
    logits = model(grid_tensor).squeeze()
    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
    preds = preds.reshape(xx.shape)
    plt.contourf(xx, yy, preds, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, alpha=0.7)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary")

plt.figure(figsize=(6,5))
plot_decision_boundary(model, X_test, y_test, device)
plt.show()

# ===================== 9. 混淆矩阵 & 错分样本分析 =====================
@torch.no_grad()
def get_preds(model, loader, device):
    all_preds = []
    all_labels = []
    all_x = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).squeeze()
        pred = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(yb.numpy())
        all_x.extend(xb.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_x)

y_pred, y_true, X_test_arr = get_preds(model, test_loader, device)
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test set)")
plt.tight_layout()
plt.show()

# 提取错误样本
err_mask = y_pred != y_true
X_err = X_test_arr[err_mask]
y_err_true = y_true[err_mask]
y_err_pred = y_pred[err_mask]
print(f"\nTotal wrong samples on test set: {len(X_err)}")
print("Wrong samples coordinates:")
print(X_err[:10])
print("True label / Pred label:")
for i in range(min(10, len(X_err))):
    print(f"Sample {i}: true={y_err_true[i]}, pred={y_err_pred[i]}, pos={X_err[i]}")

# 可视化错分样本
plt.figure(figsize=(6,5))
plot_decision_boundary(model, X_test, y_test, device)
plt.scatter(X_err[:,0], X_err[:,1], marker="*", s=200, c="black", label="Misclassified")
plt.legend()
plt.show()

"""
错分样本分析：
make_moons带噪声，两类月牙交界区域样本本身重叠。模型在边界模糊、噪声大的点容易分错。
这些样本坐标落在两个月牙的中间过渡地带，特征区分度低，是模型决策边界穿越的位置。
"""

# ===================== 10. 对照实验函数（单变量控制） =====================
def run_exp(hidden_dim=64, layers=1, act=nn.ReLU, lr=1e-3, batch_size=32):
    train_ds = MoonDataset(X_train, y_train)
    val_ds = MoonDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = MLP(hidden_dim=hidden_dim, num_hidden_layers=layers, act=act).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    tr_loss_list, val_loss_list = [], []
    tr_acc_list, val_acc_list = [], []
    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, opt, device)
        va_loss, va_acc = eval_epoch(model, val_loader, loss_fn, device)
        tr_loss_list.append(tr_loss)
        val_loss_list.append(va_loss)
        tr_acc_list.append(tr_acc)
        val_acc_list.append(va_acc)
    test_loss, test_acc = eval_epoch(model, test_loader, loss_fn, device)
    return model, tr_loss_list, val_loss_list, tr_acc_list, val_acc_list, test_loss, test_acc

# 对照组1：改变隐藏层宽度 hidden_dim=16 vs baseline=64
print("\n==== Exp1: hidden_dim=16 ====")
model1, tl1, vl1, ta1, va1, tloss1, tacc1 = run_exp(hidden_dim=16)
print(f"Test acc: {tacc1:.4f}")

# 对照组2：改变激活函数 Tanh vs ReLU
print("\n==== Exp2: act=Tanh ====")
model2, tl2, vl2, ta2, va2, tloss2, tacc2 = run_exp(act=nn.Tanh)
print(f"Test acc: {tacc2:.4f}")

