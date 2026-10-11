import numpy as np

def conv2d_numpy(image: np.ndarray, kernel: np.ndarray, padding="same", stride=1):
    """
    手写2D卷积
    :param image: 输入图像数组
        灰度图：shape [H, W]
        RGB图：shape [H, W, C]
    :param kernel: 卷积核 [Kh, Kw]
    :param padding: "same" / "valid"
    :param stride: 步长
    :return: 卷积输出，和输入维度保持一致（RGB输出每个通道单独卷积）
    """
    kh, kw = kernel.shape
    # 判断是灰度还是彩色
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = 1
        image = image[..., None]  # 增加通道维度 [H,W,1]

    # 计算padding大小
    if padding == "same":
        pad_h = (kh - 1) // 2
        pad_w = (kw - 1) // 2
    elif padding == "valid":
        pad_h = 0
        pad_w = 0
    else:
        raise ValueError("padding只能是 same 或 valid")

    # 对图像边缘填充0
    img_pad = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")

    # 输出尺寸计算公式
    out_h = int(np.floor((h + 2 * pad_h - kh) / stride + 1))
    out_w = int(np.floor((w + 2 * pad_w - kw) / stride + 1))
    output = np.zeros((out_h, out_w, c), dtype=np.float32)

    # 滑动窗口遍历
    for ci in range(c):
        for y_out in range(out_h):
            for x_out in range(out_w):
                # 当前窗口在填充图上的坐标
                y_start = y_out * stride
                y_end = y_start + kh
                x_start = x_out * stride
                x_end = x_start + kw
                patch = img_pad[y_start:y_end, x_start:x_end, ci]
                # 对应位置相乘，求和
                output[y_out, x_out, ci] = np.sum(patch * kernel)

    # 如果原始是单通道灰度，去掉通道维度
    if c == 1:
        output = output.squeeze(-1)
    return output


def clip_and_convert(img_float: np.ndarray):
    """
    卷积后数值可能超出0~255，归一化裁剪，转uint8用于保存图片
    策略：先clip到0~255，再转整数
    """
    img_clipped = np.clip(img_float, 0, 255)
    return img_clipped.astype(np.uint8)


if __name__ == "__main__":
    # 简单单元测试
    test_img = np.array([
        [10,20,30],
        [40,50,60],
        [70,80,90]
    ], dtype=np.float32)
    test_kernel = np.array([
        [1,0,-1],
        [1,0,-1],
        [1,0,-1]
    ])
    res_valid = conv2d_numpy(test_img, test_kernel, padding="valid")
    res_same = conv2d_numpy(test_img, test_kernel, padding="same")
    print("valid输出：")
    print(res_valid)
    print("same输出：")
    print(res_same)

