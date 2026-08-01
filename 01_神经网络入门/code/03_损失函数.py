"""
1.3 损失函数
运行方式：python 01_神经网络入门/code/03_损失函数.py
依赖：pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def mse_loss(y_pred, y_true):
    """均方误差损失"""
    return np.mean((y_pred - y_true) ** 2)


def mae_loss(y_pred, y_true):
    """平均绝对误差损失"""
    return np.mean(np.abs(y_pred - y_true))


def binary_cross_entropy(y_pred, y_true):
    """二分类交叉熵损失"""
    # 防止 log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def categorical_cross_entropy(y_pred, y_true):
    """多分类交叉熵损失"""
    # 防止 log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.sum(y_true * np.log(y_pred))


def demo_mse():
    """演示 MSE 损失"""
    print("=" * 50)
    print("1. 均方误差（MSE）")
    print("=" * 50)

    print("\n公式：MSE = (1/n) × Σ(ŷ - y)²")
    print("用途：回归任务")

    # 例子
    y_pred = np.array([0.8, 0.5, 0.3])
    y_true = np.array([0.2, 0.2, 0.2])

    print(f"\n预测值：{y_pred}")
    print(f"真实值：{y_true}")

    # 手动计算
    errors = y_pred - y_true
    squared_errors = errors ** 2
    mse = np.mean(squared_errors)

    print(f"\n误差：{errors}")
    print(f"平方误差：{squared_errors}")
    print(f"平均平方误差：{mse:.4f}")

    # 验证
    mse_func = mse_loss(y_pred, y_true)
    print(f"函数计算：{mse_func:.4f}")

    # 梯度
    print(f"\n梯度：∂MSE/∂ŷ = 2(ŷ - y) / n")
    grad = 2 * (y_pred - y_true) / len(y_pred)
    print(f"梯度值：{grad}")


def demo_mae():
    """演示 MAE 损失"""
    print("\n" + "=" * 50)
    print("2. 平均绝对误差（MAE）")
    print("=" * 50)

    print("\n公式：MAE = (1/n) × Σ|ŷ - y|")
    print("用途：回归任务")

    # 例子
    y_pred = np.array([0.8, 0.5, 0.3])
    y_true = np.array([0.2, 0.2, 0.2])

    print(f"\n预测值：{y_pred}")
    print(f"真实值：{y_true}")

    # 手动计算
    errors = np.abs(y_pred - y_true)
    mae = np.mean(errors)

    print(f"绝对误差：{errors}")
    print(f"平均绝对误差：{mae:.4f}")

    # 与 MSE 对比
    print(f"\n与 MSE 对比：")
    mse = mse_loss(y_pred, y_true)
    print(f"  MSE = {mse:.4f}")
    print(f"  MAE = {mae:.4f}")
    print(f"  MSE 对大误差惩罚更重（因为平方）")


def demo_binary_cross_entropy():
    """演示二分类交叉熵"""
    print("\n" + "=" * 50)
    print("3. 二分类交叉熵")
    print("=" * 50)

    print("\n公式：L = -[y × log(ŷ) + (1-y) × log(1-ŷ)]")
    print("用途：二分类任务")

    # 例子1：预测准确
    y_true = 1
    y_pred = 0.9
    loss = binary_cross_entropy(np.array([y_pred]), np.array([y_true]))

    print(f"\n例子1：预测准确")
    print(f"  真实标签：{y_true}（是猫）")
    print(f"  预测概率：{y_pred}（90% 可能是猫）")
    print(f"  损失：{loss:.4f}")

    # 例子2：预测不准
    y_pred2 = 0.1
    loss2 = binary_cross_entropy(np.array([y_pred2]), np.array([y_true]))

    print(f"\n例子2：预测不准")
    print(f"  真实标签：{y_true}（是猫）")
    print(f"  预测概率：{y_pred2}（10% 可能是猫）")
    print(f"  损失：{loss2:.4f}")

    print(f"\n解读：预测越准，损失越小")


def demo_categorical_cross_entropy():
    """演示多分类交叉熵"""
    print("\n" + "=" * 50)
    print("4. 多分类交叉熵")
    print("=" * 50)

    print("\n公式：L = -Σ yᵢ × log(ŷᵢ)")
    print("用途：多分类任务")

    # 例子
    y_true = np.array([0, 1, 0])  # one-hot 编码，是第2类
    y_pred = np.array([0.1, 0.7, 0.2])  # softmax 输出

    print(f"\n真实标签（one-hot）：{y_true}")
    print(f"预测概率：{y_pred}")

    # 手动计算
    loss = -np.sum(y_true * np.log(y_pred))
    print(f"\n计算过程：")
    print(f"  -[0×log(0.1) + 1×log(0.7) + 0×log(0.2)]")
    print(f"  = -log(0.7)")
    print(f"  = {loss:.4f}")

    # 验证
    loss_func = categorical_cross_entropy(y_pred, y_true)
    print(f"函数计算：{loss_func:.4f}")


def demo_kl_divergence():
    """演示 KL 散度"""
    print("\n" + "=" * 50)
    print("5. KL 散度")
    print("=" * 50)

    print("\n公式：KL(p||q) = Σ p(x) × log(p(x)/q(x))")
    print("用途：衡量两个分布的差异")

    # 例子
    p = np.array([0.5, 0.3, 0.2])  # 真实分布
    q = np.array([0.4, 0.4, 0.2])  # 预测分布

    print(f"\n真实分布 p：{p}")
    print(f"预测分布 q：{q}")

    # 计算 KL 散度
    kl = np.sum(p * np.log(p / q))
    print(f"\nKL(p||q) = {kl:.4f}")

    # 验证性质
    print(f"\n性质：")
    print(f"  KL ≥ 0：{kl >= 0}")
    print(f"  KL = 0 当且仅当 p = q")

    # 与交叉熵的关系
    print(f"\n与交叉熵的关系：")
    print(f"  H(p, q) = H(p) + KL(p||q)")
    print(f"  最小化交叉熵 = 最小化 KL 散度")


def visualize_losses():
    """可视化损失函数"""
    print("\n" + "=" * 50)
    print("6. 可视化损失函数")
    print("=" * 50)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # MSE vs MAE
    ax1 = axes[0]
    x = np.linspace(-2, 2, 100)
    mse = x ** 2
    mae = np.abs(x)
    ax1.plot(x, mse, 'b-', linewidth=2, label='MSE = x²')
    ax1.plot(x, mae, 'r--', linewidth=2, label='MAE = |x|')
    ax1.set_xlabel('误差 (ŷ - y)')
    ax1.set_ylabel('损失')
    ax1.set_title('MSE vs MAE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 二分类交叉熵
    ax2 = axes[1]
    y_pred = np.linspace(0.01, 0.99, 100)
    loss_y1 = -np.log(y_pred)  # y=1
    loss_y0 = -np.log(1 - y_pred)  # y=0
    ax2.plot(y_pred, loss_y1, 'b-', linewidth=2, label='y=1: -log(ŷ)')
    ax2.plot(y_pred, loss_y0, 'r--', linewidth=2, label='y=0: -log(1-ŷ)')
    ax2.set_xlabel('预测概率 ŷ')
    ax2.set_ylabel('损失')
    ax2.set_title('二分类交叉熵')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 多分类交叉熵
    ax3 = axes[2]
    p = np.array([0.5, 0.3, 0.2])
    q_range = np.linspace(0.01, 0.99, 100)
    kl_values = [np.sum(p * np.log(p / np.array([q, (1-q)*0.6, (1-q)*0.4]))) for q in q_range]
    ax3.plot(q_range, kl_values, 'b-', linewidth=2)
    ax3.axvline(x=0.5, color='r', linestyle='--', label='p₁=0.5')
    ax3.set_xlabel('q₁')
    ax3.set_ylabel('KL(p||q)')
    ax3.set_title('KL 散度')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('01_神经网络入门/images/loss_functions.png', dpi=150, bbox_inches='tight')
    print("可视化已保存到 01_神经网络入门/images/loss_functions.png")
    plt.close()


if __name__ == "__main__":
    print("损失函数详解\n")
    print("本演示将展示各种损失函数的计算和性质\n")

    demo_mse()
    demo_mae()
    demo_binary_cross_entropy()
    demo_categorical_cross_entropy()
    demo_kl_divergence()
    visualize_losses()

    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)
    print("\n关键要点：")
    print("1. MSE：回归任务，对大误差惩罚重")
    print("2. MAE：回归任务，对所有误差一视同仁")
    print("3. 二分类交叉熵：二分类任务")
    print("4. 多分类交叉熵：多分类任务")
    print("5. KL 散度：衡量分布差异")
    print("6. 最小化交叉熵 = 最小化 KL 散度")
