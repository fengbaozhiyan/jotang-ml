import numpy as np

# 1. 使用字典记录姓名与成绩
score_dict = {
    "张三": 85,
    "李四": 92,
    "王五": 78,
    "赵六": 96,
    "孙七": 88
}


# 2. 函数：计算平均分、找出最高分
def calc_score_stats(score_data):
    scores = list(score_data.values())
    avg = sum(scores) / len(scores)
    max_score = max(scores)
    return avg, max_score


avg_score, max_score = calc_score_stats(score_dict)
print("===== 学生成绩统计 =====")
print(f"原始成绩字典：{score_dict}")
print(f"平均分：{avg_score:.2f}")
print(f"最高分：{max_score}")

# 3. NumPy 创建矩阵，做矩阵乘法
# 矩阵A: 2行3列；矩阵B:3行2列，保证可以矩阵相乘，结果为 2×2
mat_a = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

mat_b = np.array([
    [7, 8],
    [9, 10],
    [11, 12]
])

# 矩阵乘法 np.dot 或者 @
mat_result = mat_a @ mat_b

print("\n===== NumPy 矩阵乘法 =====")
print(f"矩阵A:\n{mat_a}")
print(f"A的形状: {mat_a.shape}")
print(f"矩阵B:\n{mat_b}")
print(f"B的形状: {mat_b.shape}")
print(f"A × B 结果矩阵:\n{mat_result}")
print(f"结果矩阵形状: {mat_result.shape}")

