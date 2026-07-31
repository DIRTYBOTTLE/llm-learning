"""
线性代数基础演示
运行方式：python 00_数学基础/code/linear_algebra_basics.py
依赖：pip install numpy matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np


def vector_basics():
    """向量基础运算"""
    print("=" * 50)
    print("1. 向量基础运算")
    print("=" * 50)

    # 创建向量
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])

    print(f"向量 v1 = {v1}")
    print(f"向量 v2 = {v2}")

    # 向量加法
    print(f"\n向量加法：v1 + v2 = {v1 + v2}")

    # 数乘
    print(f"数乘：2 × v1 = {2 * v1}")

    # 向量长度（L2范数）
    length = np.linalg.norm(v1)
    print(f"\n向量长度：‖v1‖ = {length:.4f}")
    print(f"验证：√(1² + 2² + 3²) = {np.sqrt(1**2 + 2**2 + 3**2):.4f}")


def matrix_basics():
    """矩阵基础运算"""
    print("\n" + "=" * 50)
    print("2. 矩阵基础运算")
    print("=" * 50)

    # 创建矩阵
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])

    print(f"矩阵 A (2×3)：\n{A}")
    print(f"\n矩阵 B (3×2)：\n{B}")

    # 矩阵乘法
    C = A @ B  # 或 np.dot(A, B)
    print(f"\n矩阵乘法 A × B (2×2)：\n{C}")

    # 手动验证
    print("\n手动验证 C[0,0]：")
    print(f"  1×7 + 2×9 + 3×11 = {1*7 + 2*9 + 3*11}")


def transpose():
    """矩阵转置"""
    print("\n" + "=" * 50)
    print("3. 矩阵转置")
    print("=" * 50)

    A = np.array([[1, 2, 3],
                  [4, 5, 6]])

    print(f"原矩阵 A：\n{A}")
    print(f"\n转置 Aᵀ：\n{A.T}")

    # 转置的性质：(A × B)ᵀ = Bᵀ × Aᵀ
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])

    print("\n验证性质：(A × B)ᵀ = Bᵀ × Aᵀ")
    print(f"(A × B)ᵀ =\n{(A @ B).T}")
    print(f"Bᵀ × Aᵀ =\n{B.T @ A.T}")


def dot_product():
    """点积（内积）"""
    print("\n" + "=" * 50)
    print("4. 点积（内积）")
    print("=" * 50)

    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    print(f"向量 a = {a}")
    print(f"向量 b = {b}")

    # 点积
    dot = np.dot(a, b)
    print(f"\n点积 a · b = {dot}")
    print(f"验证：1×4 + 2×5 + 3×6 = {1*4 + 2*5 + 3*6}")

    # 点积的几何意义
    cos_theta = dot / (np.linalg.norm(a) * np.linalg.norm(b))
    theta = np.arccos(cos_theta)
    print(f"\n夹角 θ = {np.degrees(theta):.2f}°")


def attention_simulation():
    """模拟 Transformer 中的 Attention 计算"""
    print("\n" + "=" * 50)
    print("5. 模拟 Transformer 中的 Attention")
    print("=" * 50)

    # 假设有一个简单的输入序列（3个token，每个token是4维向量）
    # X = [token1, token2, token3]
    X = np.array([
        [1, 0, 1, 0],  # token 1
        [0, 2, 0, 2],  # token 2
        [1, 1, 1, 1],  # token 3
    ])

    print("输入序列 X（3个token，4维）：\n", X)

    # 定义权重矩阵（实际训练中这些是可学习的参数）
    W_Q = np.random.randn(4, 3)  # Query 权重
    W_K = np.random.randn(4, 3)  # Key 权重
    W_V = np.random.randn(4, 3)  # Value 权重

    # 计算 Q, K, V
    Q = X @ W_Q  # (3, 4) × (4, 3) = (3, 3)
    K = X @ W_K
    V = X @ W_V

    print("\nQuery Q =\n", Q)
    print("\nKey K =\n", K)
    print("\nValue V =\n", V)

    # 计算注意力分数
    d_k = K.shape[1]  # key 的维度
    scores = Q @ K.T / np.sqrt(d_k)  # 缩放点积注意力

    print(f"\n注意力分数（除以 √{d_k}）：\n", scores)

    # Softmax 归一化
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    attention_weights = softmax(scores)
    print("\n注意力权重（Softmax 后）：\n", attention_weights)
    print("每行之和（应为1）：", attention_weights.sum(axis=-1))

    # 加权求和
    output = attention_weights @ V
    print("\nAttention 输出：\n", output)


def visualize_vectors():
    """可视化向量"""
    print("\n" + "=" * 50)
    print("6. 向量可视化")
    print("=" * 50)

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：向量加法
    ax1 = axes[0]
    ax1.set_xlim(-1, 6)
    ax1.set_ylim(-1, 6)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)

    # 画向量
    ax1.annotate('', xy=(2, 3), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.text(2.1, 3.1, 'v1', fontsize=12, color='blue')

    ax1.annotate('', xy=(5, 4), xytext=(2, 3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(5.1, 4.1, 'v2', fontsize=12, color='red')

    ax1.annotate('', xy=(7, 7), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax1.text(7.1, 7.1, 'v1+v2', fontsize=12, color='green')

    ax1.set_title('向量加法', fontsize=14)

    # 右图：点积的几何意义
    ax2 = axes[1]
    ax2.set_xlim(-1, 4)
    ax2.set_ylim(-1, 4)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)

    # 画两个向量
    ax2.annotate('', xy=(3, 1), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.text(3.1, 1.1, 'a', fontsize=12, color='blue')

    ax2.annotate('', xy=(1, 3), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.text(1.1, 3.1, 'b', fontsize=12, color='red')

    # 画夹角
    theta = np.degrees(np.arctan2(1, 3))  # a 的角度
    ax2.annotate('', xy=(1.5, 0.5), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax2.text(0.8, 0.2, f'θ={np.degrees(np.arccos(np.dot([3,1], [1,3])/(np.linalg.norm([3,1])*np.linalg.norm([1,3])))):.1f}°',
             fontsize=10, color='gray')

    ax2.set_title('点积与夹角', fontsize=14)

    plt.tight_layout()
    plt.savefig('00_数学基础/code/vectors_visualization.png', dpi=150, bbox_inches='tight')
    print("可视化已保存到 vectors_visualization.png")
    plt.show()


if __name__ == "__main__":
    print("线性代数基础演示\n")
    print("本演示将展示大模型中常用的线性代数概念\n")

    vector_basics()
    matrix_basics()
    transpose()
    dot_product()
    attention_simulation()
    visualize_vectors()

    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)
    print("\n关键要点：")
    print("1. 向量是大模型的基本数据单元")
    print("2. 矩阵乘法是神经网络的核心运算")
    print("3. 点积用于计算 Attention 中的相似度")
    print("4. 转置在 Q/K/V 计算中频繁使用")
