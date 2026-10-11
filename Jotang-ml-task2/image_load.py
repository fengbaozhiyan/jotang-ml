from PIL import Image
import numpy as np
import torch

def main():
    # 1. 读取图片
    img = Image.open("test.jpg").convert("RGB")
    print("原始Pillow图像模式：", img.mode)

    # 转为 numpy HWC 格式 (高度,宽度,通道)
    img_np_hwc = np.array(img)
    print("\n===== Numpy HWC (Pillow默认) =====")
    print(f"shape: {img_np_hwc.shape}  -> [H, W, C]")
    print(f"dtype: {img_np_hwc.dtype}")
    print(f"像素最小值: {img_np_hwc.min()}, 最大值: {img_np_hwc.max()}")

    # 取一个像素点，例如 y=10, x=20
    y, x = 10, 20
    pixel_rgb = img_np_hwc[y, x]
    print(f"像素点(y={y},x={x}) RGB值 R:{pixel_rgb[0]}, G:{pixel_rgb[1]}, B:{pixel_rgb[2]}")

    # HWC -> CHW （PyTorch标准格式）
    img_np_chw = np.transpose(img_np_hwc, (2, 0, 1))
    print("\n===== Numpy CHW (PyTorch使用) =====")
    print(f"shape: {img_np_chw.shape} -> [C, H, W]")

    # 转torch tensor
    img_tensor = torch.from_numpy(img_np_chw)
    print("\n===== PyTorch Tensor =====")
    print(f"tensor shape: {img_tensor.shape}")
    print(f"tensor dtype: {img_tensor.dtype}")

if __name__ == "__main__":
    main()

