import numpy as np
import torch
import torch.nn as nn

# --------------------------
# 超参数 & 初始化（单样本）
# --------------------------
n_in = 2      # 输入维度
n_hidden = 3  # 隐藏神经元
n_out = 1     # 输出

# 输入 X shape: (1, 2)，标签 y shape: (1,1)
X_np = np.array([[0.2, 0.5]])
y_np = np.array([[1.0]])

# 参数初始化，固定随机种子保证两边一致
np.random.seed(42)
W1_np = np.random.randn(n_in, n_hidden) * 0.1
b1_np = np.zeros((1, n_hidden))
W2_np = np.random.randn(n_hidden, n_out) * 0.1
b2_np = np.zeros((1, n_out))

print("===== 参数形状 =====")
print(f"X: {X_np.shape}, y:{y_np.shape}")
print(f"W1:{W1_np.shape}, b1:{b1_np.shape}")
print(f"W2:{W2_np.shape}, b2:{b2_np.shape}")

# --------------------------
# 前向传播 Numpy
# --------------------------
def sigmoid(z):
    return 1/(1+np.exp(-z))

def relu(z):
    return np.maximum(0, z)

Z1_np = X_np @ W1_np + b1_np
A1_np = relu(Z1_np)
Z2_np = A1_np @ W2_np + b2_np
y_hat_np = sigmoid(Z2_np)
# BCE loss
loss_np = - np.mean(y_np * np.log(y_hat_np + 1e-8) + (1-y_np)*np.log(1-y_hat_np +1e-8))

print("\n===== Numpy前向结果 =====")
print(f"Z1: {Z1_np}")
print(f"A1(ReLU): {A1_np}")
print(f"Z2: {Z2_np}")
print(f"y_hat: {y_hat_np}")
print(f"loss: {loss_np}")

# --------------------------
# 反向传播 Numpy 链式法则
# --------------------------
# dL/dyhat
dL_dyhat = - y_np/(y_hat_np+1e-8) + (1-y_np)/(1-y_hat_np+1e-8)
# dL/dZ2
dL_dZ2 = dL_dyhat * y_hat_np * (1 - y_hat_np)
# dL/dW2, db2
dL_dW2 = A1_np.T @ dL_dZ2
dL_db2 = np.sum(dL_dZ2, axis=0, keepdims=True)
# dL/dA1
dL_dA1 = dL_dZ2 @ W2_np.T
# ReLU梯度：大于0导数为1
dA1_dZ1 = (Z1_np > 0).astype(float)
dL_dZ1 = dL_dA1 * dA1_dZ1
# dL/dW1, db1
dL_dW1 = X_np.T @ dL_dZ1
dL_db1 = np.sum(dL_dZ1, axis=0, keepdims=True)

print("\n===== Numpy手动梯度 =====")
print(f"dL/dW1:\n{dL_dW1}")
print(f"dL/db1:\n{dL_db1}")
print(f"dL/dW2:\n{dL_dW2}")
print(f"dL/db2:\n{dL_db2}")

# --------------------------
# PyTorch Autograd 校验，使用完全相同参数
# --------------------------
X_torch = torch.tensor(X_np, dtype=torch.float32)
y_torch = torch.tensor(y_np, dtype=torch.float32)

W1_t = torch.tensor(W1_np, dtype=torch.float32, requires_grad=True)
b1_t = torch.tensor(b1_np, dtype=torch.float32, requires_grad=True)
W2_t = torch.tensor(W2_np, dtype=torch.float32, requires_grad=True)
b2_t = torch.tensor(b2_np, dtype=torch.float32, requires_grad=True)

Z1_t = X_torch @ W1_t + b1_t
A1_t = torch.relu(Z1_t)
Z2_t = A1_t @ W2_t + b2_t
y_hat_t = torch.sigmoid(Z2_t)
loss_t = nn.BCELoss()(y_hat_t, y_torch)

loss_t.backward()

print("\n===== PyTorch autograd梯度 =====")
print(f"dL/dW1:\n{W1_t.grad.numpy()}")
print(f"dL/db1:\n{b1_t.grad.numpy()}")
print(f"dL/dW2:\n{W2_t.grad.numpy()}")
print(f"dL/db2:\n{b2_t.grad.numpy()}")

# --------------------------
# 误差对比
# --------------------------
err_W1 = np.max(np.abs(dL_dW1 - W1_t.grad.numpy()))
err_b1 = np.max(np.abs(dL_db1 - b1_t.grad.numpy()))
err_W2 = np.max(np.abs(dL_dW2 - W2_t.grad.numpy()))
err_b2 = np.max(np.abs(dL_db2 - b2_t.grad.numpy()))
print("\n===== 最大绝对误差对比 =====")
print(f"W1 max err: {err_W1:.2e}")
print(f"b1 max err: {err_b1:.2e}")
print(f"W2 max err: {err_W2:.2e}")
print(f"b2 max err: {err_b2:.2e}")

