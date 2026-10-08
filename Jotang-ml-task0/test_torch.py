import sys
import torch
import numpy
import matplotlib
import sklearn

print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"Numpy版本: {numpy.__version__}")
print(f"Matplotlib版本: {matplotlib.__version__}")
print(f"Scikit-learn版本: {sklearn.__version__}")

# PyTorch张量运算
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
c = torch.matmul(a, b)
print("\n张量a：")
print(a)
print("张量b：")
print(b)
print("矩阵乘法 a @ b：")
print(c)

# 检查GPU可用性（VMware普通虚拟机这里一定输出False，属于正常）
print("\ntorch.cuda.is_available() =", torch.cuda.is_available())

