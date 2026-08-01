"""
1.4 梯度下降与反向传播
运行方式：python 01_神经网络入门/code/04_梯度下降与反向传播.py
依赖：pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """Sigmoid 的导数"""
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    """ReLU 激活函数"""
    return np.maximum(0, z)


def relu_derivative(z):
    """ReLU 的导数"""
    return (z > 0).astype(float)


def demo_gradient():
    """演示梯度的概念"""
    print("=" * 50)
    print("1. 梯度的概念")
    print("=" * 50)

    print("\n梯度 = 损失的变化量 / 参数的变化量")
    print("∂L/∂w = (L(w+Δw) - L(w)) / Δw")

    # 例子
    w = 2.0
    L_w = 0.5

    # 尝试 w + Δw
    delta_w = 0.01
    w_new = w + delta_w
    L_w_new = 0.49  # 假设损失变小了

    gradient = (L_w_new - L_w) / delta_w

    print(f"\n例子：")
    print(f"  当前 w = {w}, 损失 L = {L_w}")
    print(f"  尝试 w = {w_new}, 损失 L = {L_w_new}")
    print(f"  梯度 = ({L_w_new} - {L_w}) / {delta_w} = {gradient}")
    print(f"\n解读：梯度是负数，说明增大 w 会让损失减小")


def demo_gradient_descent():
    """演示梯度下降"""
    print("\n" + "=" * 50)
    print("2. 梯度下降算法")
    print("=" * 50)

    print("\n公式：w_new = w_old - η × ∂L/∂w")
    print("其中 η 是学习率")

    # 目标：找到使 f(x) = (x-3)² 最小的 x
    x = 0.0  # 初始值
    learning_rate = 0.3

    print(f"\n目标：找到使 f(x) = (x-3)² 最小的 x")
    print(f"理论最小值：x = 3")
    print(f"初始值：x = {x}")
    print(f"学习率：{learning_rate}")

    print(f"\n迭代过程：")
    for step in range(10):
        # 计算损失
        loss = (x - 3) ** 2

        # 计算梯度：df/dx = 2(x-3)
        gradient = 2 * (x - 3)

        # 更新参数
        x_new = x - learning_rate * gradient

        print(f"  步骤 {step+1:2d}: x = {x:.4f}, 损失 = {loss:.4f}, 梯度 = {gradient:.4f}, x_new = {x_new:.4f}")

        x = x_new

    print(f"\n最终结果：x = {x:.4f}（接近理论值 3）")


def demo_learning_rate():
    """演示学习率的影响"""
    print("\n" + "=" * 50)
    print("3. 学习率的影响")
    print("=" * 50)

    print("\n学习率太大：来回震荡")
    print("学习率太小：收敛太慢")

    # 目标函数：f(x) = (x-3)²
    def f(x):
        return (x - 3) ** 2

    def gradient(x):
        return 2 * (x - 3)

    # 不同学习率
    learning_rates = [0.01, 0.1, 0.5, 0.9]
    x_init = 0.0
    num_steps = 20

    print(f"\n目标函数：f(x) = (x-3)²")
    print(f"初始值：x = {x_init}")
    print(f"迭代次数：{num_steps}")

    results = {}
    for lr in learning_rates:
        x = x_init
        trajectory = [x]
        for _ in range(num_steps):
            grad = gradient(x)
            x = x - lr * grad
            trajectory.append(x)
        results[lr] = trajectory
        print(f"  学习率 {lr:4.2f}: 最终 x = {x:.4f}")

    # 可视化
    plt.figure(figsize=(10, 6))

    # 画函数
    x_plot = np.linspace(-1, 7, 100)
    y_plot = f(x_plot)
    plt.plot(x_plot, y_plot, 'k-', linewidth=2, label='f(x) = (x-3)²')

    # 画轨迹
    colors = ['b', 'r', 'g', 'm']
    for i, (lr, trajectory) in enumerate(results.items()):
        plt.plot(trajectory, [f(x) for x in trajectory], f'{colors[i]}o-', 
                markersize=4, linewidth=1.5, label=f'lr={lr}')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('学习率对梯度下降的影响')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('01_神经网络入门/images/learning_rate.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存到 01_神经网络入门/images/learning_rate.png")
    plt.close()


def demo_chain_rule():
    """演示链式法则"""
    print("\n" + "=" * 50)
    print("4. 链式法则")
    print("=" * 50)

    print("\n公式：如果 y = f(g(x))，那么 dy/dx = f'(g(x)) × g'(x)")

    # 例子：y = (2x + 1)³
    print("\n例子：y = (2x + 1)³")

    x = 2.0

    # 内层函数：u = 2x + 1
    u = 2 * x + 1
    du_dx = 2

    # 外层函数：y = u³
    y = u ** 3
    dy_du = 3 * u ** 2

    # 链式法则：dy/dx = dy/du × du/dx
    dy_dx = dy_du * du_dx

    print(f"\n计算过程：")
    print(f"  x = {x}")
    print(f"  u = 2x + 1 = {u}")
    print(f"  y = u³ = {y}")
    print(f"  du/dx = 2")
    print(f"  dy/du = 3u² = {dy_du}")
    print(f"  dy/dx = dy/du × du/dx = {dy_du} × {du_dx} = {dy_dx}")

    # 验证
    def f(x):
        return (2 * x + 1) ** 3

    def numerical_derivative(f, x, h=1e-7):
        return (f(x + h) - f(x - h)) / (2 * h)

    dy_dx_numerical = numerical_derivative(f, x)
    print(f"\n数值验证：dy/dx ≈ {dy_dx_numerical:.4f}")
    print(f"解析解：dy/dx = {dy_dx:.4f}")


def demo_backpropagation():
    """演示反向传播"""
    print("\n" + "=" * 50)
    print("5. 反向传播算法")
    print("=" * 50)

    print("\n网络结构：x → [×w] → [×2] → y")
    print("即：y = 2 × (w × x)")

    # 参数
    x = 3.0
    w = 1.0
    y_true = 10.0

    print(f"\n参数：x = {x}, w = {w}, 真实值 = {y_true}")

    # 前向传播
    z1 = w * x      # z1 = w × x
    y = 2 * z1      # y = 2 × z1

    print(f"\n前向传播：")
    print(f"  z1 = w × x = {w} × {x} = {z1}")
    print(f"  y = 2 × z1 = 2 × {z1} = {y}")

    # 计算损失
    loss = (y - y_true) ** 2
    print(f"\n损失：L = (y - y_true)² = ({y} - {y_true})² = {loss}")

    # 反向传播（链式法则）
    print(f"\n反向传播（链式法则）：")

    # ∂L/∂y = 2(y - y_true)
    dL_dy = 2 * (y - y_true)
    print(f"  ∂L/∂y = 2(y - y_true) = 2({y} - {y_true}) = {dL_dy}")

    # ∂y/∂z1 = 2
    dy_dz1 = 2
    print(f"  ∂y/∂z1 = 2")

    # ∂z1/∂w = x
    dz1_dw = x
    print(f"  ∂z1/∂w = x = {x}")

    # 链式法则：∂L/∂w = ∂L/∂y × ∂y/∂z1 × ∂z1/∂w
    dL_dw = dL_dy * dy_dz1 * dz1_dw
    print(f"  ∂L/∂w = ∂L/∂y × ∂y/∂z1 × ∂z1/∂w = {dL_dy} × {dy_dz1} × {dz1_dw} = {dL_dw}")

    print(f"\n解读：梯度是 {dL_dw}，说明应该{'增大' if dL_dw < 0 else '减小'} w")


def demo_two_layer_backprop():
    """演示两层网络的反向传播（完整推导）"""
    print("\n" + "=" * 50)
    print("6. 两层网络的反向传播")
    print("=" * 50)

    print("\n网络结构：x → [线性] → [ReLU] → [线性] → y")
    print("\n前向传播公式：")
    print("  z1 = W1·x + b1      (第一层线性变换)")
    print("  a1 = ReLU(z1)       (激活函数)")
    print("  z2 = W2·a1 + b2     (第二层线性变换)")
    print("  y_pred = z2          (输出，无激活)")
    print("\n损失函数：")
    print("  L = (1/2)(y_pred - y_true)²")
    print("  为什么加 1/2？让梯度更简洁：")
    print("    如果 L = (y_pred - y_true)²，∂L/∂y_pred = 2(y_pred - y_true)")
    print("    如果 L = (1/2)(y_pred - y_true)²，∂L/∂y_pred = (y_pred - y_true)")

    # 参数
    x = np.array([1.0])
    W1 = np.array([[0.5]])  # 第一层权重
    b1 = np.array([0.1])    # 第一层偏置
    W2 = np.array([[0.8]])  # 第二层权重
    b2 = np.array([0.2])    # 第二层偏置
    y_true = np.array([2.0])

    print(f"\n参数：")
    print(f"  x = {x}, W1 = {W1}, b1 = {b1}")
    print(f"  W2 = {W2}, b2 = {b2}")
    print(f"  真实值 y = {y_true}")

    # 前向传播
    z1 = W1 @ x + b1        # 第一层线性变换
    a1 = relu(z1)            # ReLU 激活
    z2 = W2 @ a1 + b2        # 第二层线性变换
    y_pred = z2              # 输出（无激活）

    print(f"\n前向传播：")
    print(f"  z1 = W1·x + b1 = {z1}")
    print(f"  a1 = ReLU(z1) = {a1}")
    print(f"  z2 = W2·a1 + b2 = {z2}")
    print(f"  y_pred = z2 = {y_pred}")

    # 计算损失（带 1/2）
    loss = 0.5 * np.mean((y_pred - y_true) ** 2)
    print(f"\n损失：L = (1/2)(y_pred - y_true)² = {loss:.4f}")

    # 反向传播（完整推导）
    print(f"\n反向传播（完整推导）：")
    print(f"  目标：计算 ∂L/∂W1, ∂L/∂b1, ∂L/∂W2, ∂L/∂b2")
    print(f"  方法：链式法则，从输出层往前推")

    # 第一步：∂L/∂y_pred
    print(f"\n  --- 第一步：∂L/∂y_pred ---")
    print(f"  L = (1/2)(y_pred - y_true)²")
    print(f"  ∂L/∂y_pred = (y_pred - y_true)  ← 1/2 和 2 抵消了")
    dL_dy = (y_pred - y_true)
    print(f"  ∂L/∂y_pred = ({y_pred} - {y_true}) = {dL_dy}")

    # 第二步：∂L/∂z2（因为 y_pred = z2）
    print(f"\n  --- 第二步：∂L/∂z2 ---")
    print(f"  y_pred = z2，所以 ∂y_pred/∂z2 = 1")
    print(f"  ∂L/∂z2 = ∂L/∂y_pred × ∂y_pred/∂z2 = {dL_dy} × 1 = {dL_dy}")
    dL_dz2 = dL_dy

    # 第三步：∂L/∂W2 和 ∂L/∂b2
    print(f"\n  --- 第三步：∂L/∂W2 和 ∂L/∂b2 ---")
    print(f"  z2 = W2·a1 + b2")
    print(f"  ∂z2/∂W2 = a1 = {a1}")
    print(f"  ∂z2/∂b2 = 1")
    dL_dW2 = dL_dz2 * a1
    dL_db2 = dL_dz2
    print(f"  ∂L/∂W2 = ∂L/∂z2 × ∂z2/∂W2 = {dL_dz2} × {a1} = {dL_dW2}")
    print(f"  ∂L/∂b2 = ∂L/∂z2 × ∂z2/∂b2 = {dL_dz2} × 1 = {dL_db2}")

    # 第四步：∂L/∂a1（关键步骤！）
    print(f"\n  --- 第四步：∂L/∂a1（关键！）---")
    print(f"  z2 = W2·a1 + b2")
    print(f"  ∂z2/∂a1 = W2 = {W2}")
    print(f"  ∂L/∂a1 = ∂L/∂z2 × ∂z2/∂a1 = {dL_dz2} × {W2}")
    dL_da1 = dL_dz2 * W2
    print(f"  ∂L/∂a1 = {dL_da1}")

    # 第五步：∂L/∂z1
    print(f"\n  --- 第五步：∂L/∂z1 ---")
    print(f"  a1 = ReLU(z1)")
    print(f"  ∂a1/∂z1 = ReLU'(z1) = {relu_derivative(z1)}")
    print(f"  注意：ReLU'(z) = 1 if z > 0, else 0")
    da1_dz1 = relu_derivative(z1)
    dL_dz1 = dL_da1 * da1_dz1
    print(f"  ∂L/∂z1 = ∂L/∂a1 × ∂a1/∂z1 = {dL_da1} × {da1_dz1} = {dL_dz1}")

    # 第六步：∂L/∂W1 和 ∂L/∂b1
    print(f"\n  --- 第六步：∂L/∂W1 和 ∂L/∂b1 ---")
    print(f"  z1 = W1·x + b1")
    print(f"  ∂z1/∂W1 = x = {x}")
    print(f"  ∂z1/∂b1 = 1")
    dL_dW1 = dL_dz1 * x
    dL_db1 = dL_dz1
    print(f"  ∂L/∂W1 = ∂L/∂z1 × ∂z1/∂W1 = {dL_dz1} × {x} = {dL_dW1}")
    print(f"  ∂L/∂b1 = ∂L/∂z1 × ∂z1/∂b1 = {dL_dz1} × 1 = {dL_db1}")

    # 总结
    print(f"\n" + "=" * 50)
    print(f"总结：所有梯度")
    print(f"=" * 50)
    print(f"  ∂L/∂W2 = {dL_dW2}")
    print(f"  ∂L/∂b2 = {dL_db2}")
    print(f"  ∂L/∂W1 = {dL_dW1}")
    print(f"  ∂L/∂b1 = {dL_db1}")

    print(f"\n关键洞察：")
    print(f"  1. 反向传播就是链式法则的系统应用")
    print(f"  2. 从输出层开始，一层层往前算梯度")
    print(f"  3. 每一步都复用前一步的计算结果")
    print(f"  4. ∂L/∂a1 是连接两层梯度的桥梁！")


if __name__ == "__main__":
    print("梯度下降与反向传播\n")
    print("本演示将展示梯度下降和反向传播的数学原理\n")

    demo_gradient()
    demo_gradient_descent()
    demo_learning_rate()
    demo_chain_rule()
    demo_backpropagation()
    demo_two_layer_backprop()

    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)
    print("\n关键要点：")
    print("1. 梯度告诉我们要往哪个方向调整参数")
    print("2. 梯度下降：w = w - η × ∂L/∂w")
    print("3. 学习率太大震荡，太小收敛慢")
    print("4. 链式法则是反向传播的数学基础")
    print("5. 反向传播高效计算所有参数的梯度")
