import numpy as np
import torch

# ========== 1. 选择x,y值 ==========
x = np.array(2.0)
y = np.array(3.0)
print(f"x={x}, y={y}")

# ========== 前向传播 numpy ==========
a = x * y
b = x ** 2
c = a + b
f = c ** 2
print(f"Forward: a={a}, b={b}, c={c}, f={f}")

# ========== 反向传播 numpy（链式法则从后往前） ==========
df_dc = 2 * c
dc_da = 1
dc_db = 1
da_dx = y
da_dy = x
db_dx = 2 * x

df_dx = df_dc * dc_da * da_dx + df_dc * dc_db * db_dx
df_dy = df_dc * dc_da * da_dy
print(f"\nNumpy手动梯度: df/dx={df_dx}, df/dy={df_dy}")

# ========== PyTorch autograd校验 ==========
x_torch = torch.tensor(2.0, requires_grad=True)
y_torch = torch.tensor(3.0, requires_grad=True)
a_t = x_torch * y_torch
b_t = x_torch ** 2
c_t = a_t + b_t
f_t = c_t ** 2
f_t.backward()
print(f"PyTorch autograd梯度: df/dx={x_torch.grad.item()}, df/dy={y_torch.grad.item()}")

# 误差检查
err_x = abs(df_dx - x_torch.grad.item())
err_y = abs(df_dy - y_torch.grad.item())
print(f"\n最大绝对误差: dx_err={err_x}, dy_err={err_y}")

