"""
1.1 神经网络是什么
运行方式：python 01_神经网络入门/code/01_神经网络是什么.py
依赖：pip install numpy
"""

import numpy as np


def sigmoid(z):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-z))


def demo_single_neuron():
    """演示单个神经元"""
    print("=" * 50)
    print("1. 单个神经元")
    print("=" * 50)

    print("\n公式：z = wᵀx + b, y = f(z)")

    # 带伞决策的例子
    print("\n例子：决定要不要带伞")
    print("输入：x = [乌云, 天气预报, 上次经历] = [1, 1, 1]")
    print("权重：w = [0.5, 0.3, 0.4]")
    print("偏置：b = -0.5")

    x = np.array([1, 1, 1])
    w = np.array([0.5, 0.3, 0.4])
    b = -0.5

    # 计算
    z = np.dot(w, x) + b
    y = sigmoid(z)

    print(f"\n计算过程：")
    print(f"  z = wᵀx + b = 0.5×1 + 0.3×1 + 0.4×1 + (-0.5) = {z}")
    print(f"  y = sigmoid({z}) = {y:.4f}")
    print(f"\n结论：{y:.4f} > 0.5 → 带伞！")


def demo_multiple_neurons():
    """演示多个神经元"""
    print("\n" + "=" * 50)
    print("2. 多个神经元（全连接层）")
    print("=" * 50)

    print("\n同一层的神经元都接收相同的输入，但有不同的权重")

    # 输入
    x = np.array([1, 1, 0])

    # 两个神经元的权重（每行一个）
    W = np.array([
        [0.4, -0.3, -0.2],  # 神经元1
        [0.3, 0.5, 0.1]     # 神经元2
    ])
    b = np.array([0.1, -0.2])

    print(f"\n输入：x = {x}")
    print(f"权重矩阵 W（每行是一个神经元）：")
    print(f"  {W}")
    print(f"偏置：b = {b}")

    # 计算
    z = np.dot(W, x) + b
    a = sigmoid(z)

    print(f"\n计算：z = Wx + b = {z}")
    print(f"激活：a = sigmoid(z) = {a}")

    print(f"\n分开验证：")
    print(f"  神经元1：z₁ = {W[0]} · {x} + {b[0]} = {z[0]}")
    print(f"  神经元2：z₂ = {W[1]} · {x} + {b[1]} = {z[1]}")


def demo_forward_propagation():
    """演示前向传播"""
    print("\n" + "=" * 50)
    print("3. 前向传播（两层网络）")
    print("=" * 50)

    print("\n网络结构：输入(1) → 隐藏(2) → 输出(1)")

    # 输入
    x = np.array([1.0])

    # 第一层：输入(1) → 隐藏(2)
    W1 = np.array([[0.2],   # 神经元1的权重
                   [0.3]])  # 神经元2的权重
    b1 = np.array([0.1, 0.2])

    # 第二层：隐藏(2) → 输出(1)
    W2 = np.array([[0.6, 0.4]])  # 输出神经元的权重
    b2 = np.array([0.05])

    print(f"\n输入：x = {x}")
    print(f"\n第一层参数：")
    print(f"  W1 = \n{W1}")
    print(f"  b1 = {b1}")

    # 前向传播
    z1 = np.dot(W1, x) + b1
    a1 = sigmoid(z1)

    print(f"\n第一层计算：")
    print(f"  z1 = W1·x + b1 = {z1}")
    print(f"  a1 = sigmoid(z1) = {a1}")

    print(f"\n第二层参数：")
    print(f"  W2 = \n{W2}")
    print(f"  b2 = {b2}")

    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)

    print(f"\n第二层计算：")
    print(f"  z2 = W2·a1 + b2 = {z2[0]:.4f}")
    print(f"  a2 = sigmoid(z2) = {a2[0]:.4f}")

    print(f"\n最终输出：{a2[0]:.4f}")


def demo_complete_example():
    """完整例子：判断要不要出去玩"""
    print("\n" + "=" * 50)
    print("4. 完整例子：判断要不要出去玩")
    print("=" * 50)

    print("\n任务：根据 3 个信息，判断今天要不要出去玩")
    print("输入：x = [是否周末, 天气晴朗, 朋友约]")
    print("输出：是否出去玩")

    # 输入
    x = np.array([1, 1, 0])  # 周末，晴朗，没人约

    # 隐藏层参数
    W1 = np.array([
        [0.4, -0.3, -0.2],  # 神经元1
        [0.3, 0.5, 0.1]     # 神经元2
    ])
    b1 = np.array([0.1, -0.2])

    # 输出层参数
    W2 = np.array([0.6, 0.4])
    b2 = 0.05

    print(f"\n输入：x = {x} (周末，晴朗，没人约)")
    print(f"\n隐藏层参数：")
    print(f"  神经元1：w₁ = {W1[0]}, b₁ = {b1[0]}")
    print(f"  神经元2：w₂ = {W1[1]}, b₂ = {b1[1]}")
    print(f"\n输出层参数：")
    print(f"  w₃ = {W2}, b₃ = {b2}")

    # 隐藏层计算
    print(f"\n隐藏层计算：")
    z1 = np.dot(W1, x) + b1
    a1 = np.maximum(0, z1)  # ReLU

    print(f"  神经元1：z₁ = {W1[0]} · {x} + {b1[0]} = {z1[0]}")
    print(f"  神经元2：z₂ = {W1[1]} · {x} + {b1[1]} = {z1[1]}")
    print(f"  ReLU：a₁ = {a1}")

    # 输出层计算
    print(f"\n输出层计算：")
    z2 = np.dot(W2, a1) + b2
    y = sigmoid(z2)

    print(f"  z₃ = {W2} · {a1} + {b2} = {z2:.4f}")
    print(f"  y = sigmoid({z2:.4f}) = {y:.4f}")

    print(f"\n结论：{y:.4f} > 0.5 → 可以出去玩！")


def demo_matrix_notation():
    """演示矩阵表示"""
    print("\n" + "=" * 50)
    print("5. 矩阵表示（更简洁）")
    print("=" * 50)

    print("\n用矩阵可以一次计算所有神经元：")
    print("z = Wx + b")

    # 输入
    x = np.array([1, 1, 0])

    # 权重矩阵
    W = np.array([
        [0.4, -0.3, -0.2],  # 神经元1
        [0.3, 0.5, 0.1]     # 神经元2
    ])
    b = np.array([0.1, -0.2])

    print(f"\n输入：x = {x}")
    print(f"权重矩阵 W（每行是一个神经元）：")
    print(f"  {W}")
    print(f"偏置：b = {b}")

    # 矩阵乘法
    z = np.dot(W, x) + b
    a = np.maximum(0, z)  # ReLU

    print(f"\n计算：z = Wx + b = {z}")
    print(f"激活：a = ReLU(z) = {a}")

    print(f"\n好处：")
    print(f"  1. 代码简洁")
    print(f"  2. 计算高效（NumPy 优化）")
    print(f"  3. 易于扩展到更多神经元")


if __name__ == "__main__":
    print("神经网络是什么\n")
    print("本演示将展示神经网络的基本概念\n")

    demo_single_neuron()
    demo_multiple_neurons()
    demo_forward_propagation()
    demo_complete_example()
    demo_matrix_notation()

    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)
    print("\n关键要点：")
    print("1. 神经元 = 加权求和 + 激活函数")
    print("2. 同一层的神经元接收相同输入，有不同权重")
    print("3. 前向传播 = 从输入到输出的计算")
    print("4. 矩阵乘法可以高效计算多个神经元")
