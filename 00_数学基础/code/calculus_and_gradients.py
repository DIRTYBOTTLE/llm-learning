"""
微积分与梯度演示
运行方式：python 00_数学基础/code/calculus_and_gradients.py
依赖：pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def numerical_derivative(f, x, h=1e-7):
    """数值计算导数（有限差分法）"""
    return (f(x + h) - f(x - h)) / (2 * h)


def derivative_basics():
    """导数基础"""
    print("=" * 50)
    print("1. 导数基础")
    print("=" * 50)

    # 定义几个函数
    f1 = lambda x: x ** 2  # f(x) = x²
    f2 = lambda x: np.sin(x)  # f(x) = sin(x)
    f3 = lambda x: np.exp(x)  # f(x) = eˣ

    # 计算导数
    x = 2.0
    print(f"在 x = {x} 处：")
    print(f"  f(x) = x²:     f'({x}) = {numerical_derivative(f1, x):.4f} (理论值: {2*x})")
    print(f"  f(x) = sin(x): f'({x}) = {numerical_derivative(f2, x):.4f} (理论值: {np.cos(x):.4f})")
    print(f"  f(x) = eˣ:     f'({x}) = {numerical_derivative(f3, x):.4f} (理论值: {np.exp(x):.4f})")


def chain_rule():
    """链式法则演示"""
    print("\n" + "=" * 50)
    print("2. 链式法则")
    print("=" * 50)

    # 例1: y = (2x + 1)³
    # 外层函数: u³, 导数 3u²
    # 内层函数: 2x + 1, 导数 2
    # 链式法则: dy/dx = 3(2x+1)² × 2

    def f(x):
        return (2 * x + 1) ** 3

    def df_dx(x):
        # 链式法则计算
        u = 2 * x + 1
        return 3 * u ** 2 * 2

    x = 1.0
    numerical = numerical_derivative(f, x)
    analytical = df_dx(x)

    print(f"函数: y = (2x + 1)³")
    print(f"在 x = {x} 处：")
    print(f"  数值导数: {numerical:.6f}")
    print(f"  解析导数: {analytical:.6f}")
    print(f"  差异: {abs(numerical - analytical):.10f}")

    # 例2: y = sin(x²)
    def g(x):
        return np.sin(x ** 2)

    def dg_dx(x):
        # 链式法则: cos(x²) × 2x
        return np.cos(x ** 2) * 2 * x

    x = 1.5
    numerical = numerical_derivative(g, x)
    analytical = dg_dx(x)

    print(f"\n函数: y = sin(x²)")
    print(f"在 x = {x} 处：")
    print(f"  数值导数: {numerical:.6f}")
    print(f"  解析导数: {analytical:.6f}")


def gradient_basics():
    """梯度基础"""
    print("\n" + "=" * 50)
    print("3. 梯度计算")
    print("=" * 50)

    # 二元函数: f(x, y) = x² + y²
    def f(x, y):
        return x ** 2 + y ** 2

    # 梯度: ∇f = [2x, 2y]
    def gradient(x, y):
        return np.array([2 * x, 2 * y])

    x, y = 1.0, 2.0
    grad = gradient(x, y)
    print(f"函数: f(x, y) = x² + y²")
    print(f"在点 ({x}, {y}) 处：")
    print(f"  f({x}, {y}) = {f(x, y)}")
    print(f"  梯度 ∇f = {grad}")
    print(f"  梯度方向（单位向量）= {grad / np.linalg.norm(grad)}")


def gradient_descent_demo():
    """梯度下降演示"""
    print("\n" + "=" * 50)
    print("4. 梯度下降演示")
    print("=" * 50)

    # 目标函数: f(x, y) = (x-2)² + (y-3)²
    # 最小值在 (2, 3)
    def f(x, y):
        return (x - 2) ** 2 + (y - 3) ** 2

    def gradient(x, y):
        return np.array([2 * (x - 2), 2 * (y - 3)])

    # 初始点
    x, y = 0.0, 0.0
    learning_rate = 0.1
    num_steps = 20

    print(f"目标: 最小化 f(x, y) = (x-2)² + (y-3)²")
    print(f"理论最小值: (2, 3), f = 0")
    print(f"初始点: ({x}, {y}), f = {f(x, y):.4f}")
    print(f"学习率: {learning_rate}")
    print(f"迭代次数: {num_steps}\n")

    # 记录轨迹
    trajectory = [(x, y, f(x, y))]

    for i in range(num_steps):
        grad = gradient(x, y)
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        trajectory.append((x, y, f(x, y)))

        if i % 5 == 0 or i == num_steps - 1:
            print(f"  步骤 {i+1:2d}: x = {x:.4f}, y = {y:.4f}, f = {f(x, y):.6f}")

    print(f"\n最终结果: ({x:.4f}, {y:.4f}), f = {f(x, y):.6f}")
    return trajectory


def visualize_gradient_descent():
    """可视化梯度下降轨迹"""
    print("\n" + "=" * 50)
    print("5. 梯度下降可视化")
    print("=" * 50)

    # 运行梯度下降
    trajectory = gradient_descent_demo()

    # 创建等高线图
    x = np.linspace(-1, 4, 100)
    y = np.linspace(-1, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = (X - 2) ** 2 + (Y - 3) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：等高线图
    ax1 = axes[0]
    contour = ax1.contour(X, Y, Z, levels=20, cmap='viridis')
    ax1.clabel(contour, inline=True, fontsize=8)

    # 画轨迹
    traj_x = [t[0] for t in trajectory]
    traj_y = [t[1] for t in trajectory]
    ax1.plot(traj_x, traj_y, 'ro-', markersize=4, linewidth=1.5, label='梯度下降轨迹')
    ax1.plot(2, 3, 'g*', markersize=15, label='最小值 (2,3)')
    ax1.plot(0, 0, 'bs', markersize=10, label='起始点 (0,0)')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('梯度下降轨迹（等高线图）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图：损失曲线
    ax2 = axes[1]
    losses = [t[2] for t in trajectory]
    ax2.plot(range(len(losses)), losses, 'b-o', markersize=3)
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('损失值 f(x,y)')
    ax2.set_title('损失随迭代次数变化')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('00_数学基础/code/gradient_descent.png', dpi=150, bbox_inches='tight')
    print("可视化已保存到 gradient_descent.png")
    plt.show()


def backprop_simulation():
    """模拟简单的反向传播"""
    print("\n" + "=" * 50)
    print("6. 反向传播模拟（链式法则应用）")
    print("=" * 50)

    # 简单的计算图: z = (x * y + b)²
    # x=2, y=3, b=1
    x, y, b = 2.0, 3.0, 1.0

    # 前向传播
    w = x * y  # w = 6
    z = w + b  # z = 7
    loss = z ** 2  # loss = 49

    print(f"计算图: loss = (x × y + b)²")
    print(f"输入: x = {x}, y = {y}, b = {b}")
    print(f"\n前向传播:")
    print(f"  w = x × y = {x} × {y} = {w}")
    print(f"  z = w + b = {w} + {b} = {z}")
    print(f"  loss = z² = {z}² = {loss}")

    # 反向传播（链式法则）
    dloss_dz = 2 * z  # loss 对 z 的导数
    dz_dw = 1  # z 对 w 的导数
    dz_db = 1  # z 对 b 的导数
    dw_dx = y  # w 对 x 的导数
    dw_dy = x  # w 对 y 的导数

    dloss_dw = dloss_dz * dz_dw  # 链式法则
    dloss_db = dloss_dz * dz_db
    dloss_dx = dloss_dw * dw_dx
    dloss_dy = dloss_dw * dw_dy

    print(f"\n反向传播（链式法则）:")
    print(f"  dloss/dz = 2z = {dloss_dz}")
    print(f"  dloss/dw = dloss/dz × dz/dw = {dloss_dz} × {dz_dw} = {dloss_dw}")
    print(f"  dloss/db = dloss/dz × dz/db = {dloss_dz} × {dz_db} = {dloss_db}")
    print(f"  dloss/dx = dloss/dw × dw/dx = {dloss_dw} × {dw_dx} = {dloss_dx}")
    print(f"  dloss/dy = dloss/dw × dw/dy = {dloss_dw} × {dw_dy} = {dloss_dy}")

    # 数值验证
    h = 1e-7
    numerical_dx = (((x + h) * y + b) ** 2 - ((x - h) * y + b) ** 2) / (2 * h)
    print(f"\n数值验证: dloss/dx ≈ {numerical_dx:.6f} (解析值: {dloss_dx})")


def visualize_derivatives():
    """可视化导数"""
    print("\n" + "=" * 50)
    print("7. 导数可视化")
    print("=" * 50)

    x = np.linspace(-2, 4, 100)
    f = x ** 2
    df = 2 * x

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：函数及其导数
    ax1 = axes[0]
    ax1.plot(x, f, 'b-', linewidth=2, label='f(x) = x²')
    ax1.plot(x, df, 'r--', linewidth=2, label="f'(x) = 2x")
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('函数与导数')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图：切线演示
    ax2 = axes[1]
    ax2.plot(x, f, 'b-', linewidth=2, label='f(x) = x²')

    # 在 x=1 处画切线
    x0 = 1
    tangent_slope = 2 * x0
    tangent_x = np.array([x0 - 1, x0 + 1])
    tangent_y = f(x0) + tangent_slope * (tangent_x - x0)
    ax2.plot(tangent_x, tangent_y, 'r--', linewidth=2, label=f'x={x0} 处切线')
    ax2.plot(x0, f(x0), 'go', markersize=10, label=f'切点 ({x0}, {f(x0)})')

    ax2.set_xlim(-1, 3)
    ax2.set_ylim(-1, 5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('导数的几何意义（切线斜率）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('00_数学基础/code/derivatives_visualization.png', dpi=150, bbox_inches='tight')
    print("可视化已保存到 derivatives_visualization.png")
    plt.show()


if __name__ == "__main__":
    print("微积分与梯度演示\n")
    print("本演示将展示大模型训练中核心的微积分概念\n")

    derivative_basics()
    chain_rule()
    gradient_basics()
    gradient_descent_demo()
    visualize_gradient_descent()
    backprop_simulation()
    visualize_derivatives()

    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)
    print("\n关键要点：")
    print("1. 导数描述函数的瞬时变化率")
    print("2. 链式法则是反向传播的数学基础")
    print("3. 梯度指向函数增长最快的方向")
    print("4. 梯度下降是模型训练的核心算法")
    print("5. 反向传播就是链式法则的系统应用")
