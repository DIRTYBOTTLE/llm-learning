"""
1.2 激活函数详解
运行方式：python 01_神经网络入门/code/02_激活函数详解.py
依赖：pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """Sigmoid 的导数：σ'(z) = σ(z) × (1 - σ(z))"""
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    """ReLU 激活函数"""
    return np.maximum(0, z)


def relu_derivative(z):
    """ReLU 的导数"""
    return np.where(z > 0, 1.0, 0.0)


def tanh(z):
    """Tanh 激活函数"""
    return np.tanh(z)


def tanh_derivative(z):
    """Tanh 的导数：tanh'(z) = 1 - tanh²(z)"""
    return 1 - np.tanh(z) ** 2


def softmax(z):
    """Softmax 激活函数"""
    exp_z = np.exp(z - np.max(z))  # 减去最大值防止溢出
    return exp_z / np.sum(exp_z)


def demo_sigmoid():
    """演示 Sigmoid 函数"""
    print("=" * 50)
    print("1. Sigmoid 函数")
    print("=" * 50)

    print("\n公式：σ(z) = 1 / (1 + e^(-z))")
    print("输出范围：(0, 1)")

    # 测试几个值
    test_values = [-10, -1, 0, 1, 10]
    print("\n测试值：")
    for z in test_values:
        y = sigmoid(z)
        dy = sigmoid_derivative(z)
        print(f"  z = {z:3d}  →  σ(z) = {y:.6f}  →  σ'(z) = {dy:.6f}")

    # 验证导数公式
    print("\n验证导数公式：σ'(z) = σ(z) × (1 - σ(z))")
    z = 2.0
    s = sigmoid(z)
    ds = sigmoid_derivative(z)
    ds_formula = s * (1 - s)
    print(f"  z = {z}")
    print(f"  σ(z) = {s:.6f}")
    print(f"  σ'(z) = {ds:.6f}")
    print(f"  σ(z) × (1 - σ(z)) = {ds_formula:.6f}")
    print(f"  两者相等？{np.isclose(ds, ds_formula)}")


def demo_relu():
    """演示 ReLU 函数"""
    print("\n" + "=" * 50)
    print("2. ReLU 函数")
    print("=" * 50)

    print("\n公式：ReLU(z) = max(0, z)")
    print("输出范围：[0, ∞)")

    # 测试几个值
    test_values = [-5, -1, 0, 1, 5]
    print("\n测试值：")
    for z in test_values:
        y = relu(z)
        dy = relu_derivative(z)
        print(f"  z = {z:3d}  →  ReLU(z) = {y:.1f}  →  ReLU'(z) = {dy:.1f}")


def demo_tanh():
    """演示 Tanh 函数"""
    print("\n" + "=" * 50)
    print("3. Tanh 函数")
    print("=" * 50)

    print("\n公式：tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))")
    print("输出范围：(-1, 1)")

    # 测试几个值
    test_values = [-10, -1, 0, 1, 10]
    print("\n测试值：")
    for z in test_values:
        y = tanh(z)
        dy = tanh_derivative(z)
        print(f"  z = {z:3d}  →  tanh(z) = {y:.6f}  →  tanh'(z) = {dy:.6f}")

    # 验证与 Sigmoid 的关系
    print("\n验证：tanh(z) = 2σ(2z) - 1")
    z = 1.5
    t = tanh(z)
    s = 2 * sigmoid(2 * z) - 1
    print(f"  z = {z}")
    print(f"  tanh(z) = {t:.6f}")
    print(f"  2σ(2z) - 1 = {s:.6f}")
    print(f"  两者相等？{np.isclose(t, s)}")


def demo_softmax():
    """演示 Softmax 函数"""
    print("\n" + "=" * 50)
    print("4. Softmax 函数")
    print("=" * 50)

    print("\n公式：softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)")
    print("输出：概率分布（和为1）")

    # 测试
    z = np.array([2.0, 1.0, 0.1])
    probs = softmax(z)

    print(f"\n输入：z = {z}")
    print(f"e^z = {np.exp(z)}")
    print(f"总和：{np.sum(np.exp(z)):.4f}")
    print(f"Softmax：{probs}")
    print(f"总和：{np.sum(probs):.4f}（应该是1）")

    # 验证
    print(f"\n解读：")
    print(f"  第1类概率：{probs[0]:.4f}（{probs[0]*100:.1f}%）")
    print(f"  第2类概率：{probs[1]:.4f}（{probs[1]*100:.1f}%）")
    print(f"  第3类概率：{probs[2]:.4f}（{probs[2]*100:.1f}%）")


def demo_gradient_vanishing():
    """演示梯度消失问题"""
    print("\n" + "=" * 50)
    print("5. 梯度消失问题")
    print("=" * 50)

    print("\nSigmoid 的问题：当 z 很大或很小时，梯度接近 0")
    print("\nz 值      | σ(z)      | σ'(z)     | 问题")
    print("-" * 50)

    for z in [-10, -5, -1, 0, 1, 5, 10]:
        s = sigmoid(z)
        ds = sigmoid_derivative(z)
        problem = "梯度消失!" if ds < 0.01 else ""
        print(f"z = {z:3d}    | {s:.6f} | {ds:.6f} | {problem}")

    print("\nReLU 的优势：z > 0 时梯度恒为 1，不会消失")
    print("\nz 值      | ReLU(z)   | ReLU'(z)  | 优势")
    print("-" * 50)

    for z in [-10, -5, -1, 0, 1, 5, 10]:
        r = relu(z)
        dr = relu_derivative(z)
        advantage = "梯度恒定!" if z > 0 else ""
        print(f"z = {z:3d}    | {r:6.1f}    | {dr:6.1f}    | {advantage}")


def visualize_activations():
    """可视化激活函数"""
    print("\n" + "=" * 50)
    print("6. 可视化激活函数")
    print("=" * 50)

    z = np.linspace(-10, 10, 100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Sigmoid
    ax1 = axes[0, 0]
    ax1.plot(z, sigmoid(z), 'b-', linewidth=2, label='σ(z)')
    ax1.plot(z, sigmoid_derivative(z), 'r--', linewidth=2, label="σ'(z)")
    ax1.set_title('Sigmoid')
    ax1.set_xlabel('z')
    ax1.set_ylabel('值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    # ReLU
    ax2 = axes[0, 1]
    ax2.plot(z, relu(z), 'b-', linewidth=2, label='ReLU(z)')
    ax2.plot(z, relu_derivative(z), 'r--', linewidth=2, label="ReLU'(z)")
    ax2.set_title('ReLU')
    ax2.set_xlabel('z')
    ax2.set_ylabel('值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Tanh
    ax3 = axes[1, 0]
    ax3.plot(z, tanh(z), 'b-', linewidth=2, label='tanh(z)')
    ax3.plot(z, tanh_derivative(z), 'r--', linewidth=2, label="tanh'(z)")
    ax3.set_title('Tanh')
    ax3.set_xlabel('z')
    ax3.set_ylabel('值')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # 梯度消失对比
    ax4 = axes[1, 1]
    z_large = np.linspace(-10, 10, 100)
    ax4.plot(z_large, sigmoid_derivative(z_large), 'r-', linewidth=2, label="σ'(z)")
    ax4.plot(z_large, np.where(z_large > 0, 1, 0), 'b-', linewidth=2, label="ReLU'(z)")
    ax4.set_title('梯度对比：Sigmoid vs ReLU')
    ax4.set_xlabel('z')
    ax4.set_ylabel('梯度')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.1, 1.2)

    plt.tight_layout()
    plt.savefig('01_神经网络入门/images/activation_functions.png', dpi=150, bbox_inches='tight')
    print("可视化已保存到 01_神经网络入门/images/activation_functions.png")
    plt.close()


if __name__ == "__main__":
    print("激活函数详解\n")
    print("本演示将展示各种激活函数的性质和导数\n")

    demo_sigmoid()
    demo_relu()
    demo_tanh()
    demo_softmax()
    demo_gradient_vanishing()
    visualize_activations()

    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)
    print("\n关键要点：")
    print("1. Sigmoid：输出(0,1)，有梯度消失问题")
    print("2. ReLU：计算快，缓解梯度消失")
    print("3. Tanh：输出(-1,1)，零中心")
    print("4. Softmax：输出概率分布，用于分类")
    print("5. 选择激活函数要考虑梯度问题")
