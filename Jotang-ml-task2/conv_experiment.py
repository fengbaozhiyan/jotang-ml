from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy_conv import conv2d_numpy, clip_and_convert

def gaussian_kernel_5x5(sigma=1):
    """生成5×5高斯卷积核"""
    ax = np.arange(-2, 3)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def main():
    # 读取图片，转为HWC numpy数组 float
    img_pil = Image.open("test.jpg").convert("RGB")
    img_rgb = np.array(img_pil, dtype=np.float32)

    # ========= 定义4组卷积核 =========
    kernel_mean3 = np.ones((3,3)) / 9.0                      # 3x3均值模糊
    kernel_gauss5 = gaussian_kernel_5x5(sigma=1)             # 5x5高斯模糊
    kernel_sharp = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])   # 锐化核
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])         # Sobel X 水平边缘
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])         # Sobel Y 垂直边缘

    # ========= same padding 实验 =========
    print("正在计算 same padding 结果")
    out_mean_same = conv2d_numpy(img_rgb, kernel_mean3, padding="same")
    out_gauss_same = conv2d_numpy(img_rgb, kernel_gauss5, padding="same")
    out_sharp_same = conv2d_numpy(img_rgb, kernel_sharp, padding="same")
    out_sobelx = conv2d_numpy(img_rgb, sobel_x, padding="same")
    out_sobely = conv2d_numpy(img_rgb, sobel_y, padding="same")
    out_sobel_combine = np.sqrt(out_sobelx**2 + out_sobely**2)

    # ========= valid padding 实验 =========
    print("正在计算 valid padding 结果")
    out_mean_valid = conv2d_numpy(img_rgb, kernel_mean3, padding="valid")

    # ========= 可视化 =========
    plt.figure(figsize=(16, 9))
    plt.subplot(2,3,1)
    plt.title("Original")
    plt.imshow(img_rgb.astype(np.uint8))
    plt.axis("off")

    plt.subplot(2,3,2)
    plt.title("Mean Blur 3x3 same")
    plt.imshow(clip_and_convert(out_mean_same))
    plt.axis("off")

    plt.subplot(2,3,3)
    plt.title("Gaussian Blur 5x5 same")
    plt.imshow(clip_and_convert(out_gauss_same))
    plt.axis("off")

    plt.subplot(2,3,4)
    plt.title("Sharpen same")
    plt.imshow(clip_and_convert(out_sharp_same))
    plt.axis("off")

    plt.subplot(2,3,5)
    plt.title("Sobel combined edge")
    plt.imshow(clip_and_convert(out_sobel_combine))
    plt.axis("off")

    plt.subplot(2,3,6)
    plt.title("Mean Blur valid")
    plt.imshow(clip_and_convert(out_mean_valid))
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("conv_result.png", dpi=150)
    plt.show()

    # ========= 输出尺寸关系 =========
    print(f"\n原图HWC shape: {img_rgb.shape}")
    print(f"same均值输出shape: {out_mean_same.shape}")
    print(f"valid均值输出shape: {out_mean_valid.shape}")

if __name__ == "__main__":
    main()

