import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# ====================== 固定随机种子，保证复现 ======================
SEED = 42
np.random.seed(SEED)

# ---------------------- 1. 加载 make_moons 数据集，和Task1完全一致 ----------------------
X, y = make_moons(n_samples=1000, noise=0.2, random_state=SEED)
# 划分：test 20%，剩下80%分出train70% val10%，和Task1一致
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=SEED, stratify=y_temp)

# reshape y为(N,1)
y_train = y_train.reshape(-1,1)
y_val = y_val.reshape(-1,1)
y_test = y_test.reshape(-1,1)
print(f"Train:{X_train.shape}, Val:{X_val.shape}, Test:{X_test.shape}")

# ---------------------- 2. 网络工具函数 ----------------------
def sigmoid(z):
    # 防止溢出
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def relu(z):
    return np.maximum(z, 0.0)

def bce_loss(yhat, y):
    eps = 1e-8
    return -np.mean(y * np.log(yhat+eps) + (1-y)*np.log(1-yhat+eps))

def relu_grad(z):
    return (z>0).astype(np.float32)

# ---------------------- 3. 两层MLP类：numpy前向 + 手写BP + SGD ----------------------
class NumpyMLP:
    def __init__(self, in_dim=2, hid_dim=16, out_dim=1, lr=1e-3, seed=42):
        np.random.seed(seed)
        # 权重初始化，Xavier简易初始化
        self.W1 = np.random.randn(in_dim, hid_dim) * np.sqrt(2.0/in_dim)
        self.b1 = np.zeros((1, hid_dim))
        self.W2 = np.random.randn(hid_dim, out_dim) * np.sqrt(2.0/hid_dim)
        self.b2 = np.zeros((1, out_dim))
        self.lr = lr

    def forward(self, X):
        # X: [B,2]
        self.Z1 = X @ self.W1 + self.b1    # [B,16]
        self.H1 = relu(self.Z1)            # [B,16]
        self.Z2 = self.H1 @ self.W2 + self.b2 # [B,1]
        self.Yhat = sigmoid(self.Z2)       # [B,1]
        return self.Yhat

    def backward(self, X, y):
        B = X.shape[0]
        dZ2 = self.Yhat - y                             # [B,1]
        dW2 = (self.H1.T @ dZ2) / B                     # [16,1]
        db2 = np.sum(dZ2, axis=0, keepdims=True)/B      # [1,1]

        dH1 = dZ2 @ self.W2.T                           # [B,16]
        dZ1 = dH1 * relu_grad(self.Z1)                  # [B,16]
        dW1 = (X.T @ dZ1) / B                           # [2,16]
        db1 = np.sum(dZ1, axis=0, keepdims=True)/B      # [1,16]

        # SGD 参数更新
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X):
        yh = self.forward(X)
        return (yh>0.5).astype(np.float32)

# ---------------------- 4. 训练循环（mini-batch） ----------------------
def get_batches(X, y, batch_size, shuffle=True):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx]

model = NumpyMLP(in_dim=2, hid_dim=16, out_dim=1, lr=1e-3)
EPOCHS = 100
BATCH_SIZE = 32

train_loss_hist = []
train_acc_hist = []
val_loss_hist = []
val_acc_hist = []

for epoch in range(EPOCHS):
    # train
    model.forward(X_train)
    tr_loss = bce_loss(model.Yhat, y_train)
    tr_pred = model.predict(X_train)
    tr_acc = np.mean(tr_pred == y_train)

    # mini batch SGD更新参数
    for xb, yb in get_batches(X_train, y_train, BATCH_SIZE):
        model.forward(xb)
        model.backward(xb, yb)

    # val
    model.forward(X_val)
    va_loss = bce_loss(model.Yhat, y_val)
    va_pred = model.predict(X_val)
    va_acc = np.mean(va_pred == y_val)

    train_loss_hist.append(tr_loss)
    train_acc_hist.append(tr_acc)
    val_loss_hist.append(va_loss)
    val_acc_hist.append(va_acc)

    if (epoch+1)%10 == 0:
        print(f"Epoch {epoch+1:3d} | Train Loss:{tr_loss:.4f} Acc:{tr_acc:.4f} | Val Loss:{va_loss:.4f} Acc:{va_acc:.4f}")

# 测试集评估
model.forward(X_test)
test_loss = bce_loss(model.Yhat, y_test)
test_pred = model.predict(X_test)
test_acc = np.mean(test_pred == y_test)
print(f"\n==== NumPy手写BP模型 测试集结果 ====")
print(f"Test Loss:{test_loss:.4f}, Test Acc:{test_acc:.4f}")

# ---------------------- 绘图1：Loss & Acc曲线 ----------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_loss_hist, label="Train Loss(numpy)")
plt.plot(val_loss_hist, label="Val Loss(numpy)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Numpy-BP Loss Curve")

plt.subplot(1,2,2)
plt.plot(train_acc_hist, label="Train Acc(numpy)")
plt.plot(val_acc_hist, label="Val Acc(numpy)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Numpy-BP Acc Curve")
plt.tight_layout()
plt.show()

# ---------------------- 绘图2：决策边界 ----------------------
def plot_np_decision_boundary(model, X, y):
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    grid = np.c_[xx.ravel(), yy.ravel()]
    yh = model.forward(grid)
    Z = yh.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    plt.scatter(X[y[:,0]==0,0], X[y[:,0]==0,1], c="blue", s=30, alpha=0.7, label="Class0")
    plt.scatter(X[y[:,0]==1,0], X[y[:,0]==1,1], c="red", s=30, alpha=0.7, label="Class1")
    plt.legend()
    plt.title("Numpy手写BP MLP 决策边界")
    plt.show()

plot_np_decision_boundary(model, X_train, y_train)

