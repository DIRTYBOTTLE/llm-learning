"""
1.5 第一个神经网络
运行方式：python 01_神经网络入门/code/05_第一个神经网络.py
依赖：pip install numpy torch
"""

import numpy as np


# ============================================================
# 第 2 节：用 NumPy 手写 - 最简单的神经元
# ============================================================
def sigmoid(z):
    """Sigmoid 激活函数：把数字压缩到 0~1"""
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """Sigmoid 的导数：σ'(z) = σ(z) × (1 - σ(z))"""
    s = sigmoid(z)
    return s * (1 - s)


def demo_single_neuron():
    """演示：单个神经元"""
    print("=" * 50)
    print("2. 最简单的神经元")
    print("=" * 50)

    # 定义神经元
    def neuron(x, weight, bias):
        """单个神经元的计算"""
        z = weight * x + bias  # 加权求和
        y = sigmoid(z)         # 激活
        return y, z

    # 测试
    x = 1.0
    weight = 0.5
    bias = -0.5

    y, z = neuron(x, weight, bias)
    print(f"输入: {x}")
    print(f"权重: {weight}")
    print(f"偏置: {bias}")
    print(f"原始值 z = {z:.4f}")
    print(f"激活值 y = sigmoid({z:.4f}) = {y:.4f}")
    print(f"\n解读：z = 0.5 × 1.0 + (-0.5) = 0.0, y = sigmoid(0.0) = 0.5")


# ============================================================
# 第 3 节：两层神经网络
# ============================================================
def demo_two_layer_network():
    """演示：两层神经网络"""
    print("\n" + "=" * 50)
    print("3. 两层神经网络")
    print("=" * 50)

    # 初始化参数
    W1 = np.array([[0.2, 0.4],
                   [0.3, 0.5]])  # 输入→隐藏
    b1 = np.array([0.1, 0.2])

    W2 = np.array([[0.6],
                   [0.4]])       # 隐藏→输出
    b2 = np.array([0.05])

    # 前向传播
    def forward(x):
        z1 = np.dot(W1, x) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = sigmoid(z2)
        return a2, a1, z2, z1

    # 测试
    x = np.array([1.0])
    a2, a1, z2, z1 = forward(x)

    print(f"输入: {x}")
    print(f"\n隐藏层:")
    print(f"  z1 = W1·x + b1 = {z1}")
    print(f"  a1 = sigmoid(z1) = {a1}")
    print(f"\n输出层:")
    print(f"  z2 = W2·a1 + b2 = {z2[0]:.4f}")
    print(f"  a2 = sigmoid(z2) = {a2[0]:.4f}")


# ============================================================
# 第 4 节：加入学习 - 梯度下降
# ============================================================
def demo_gradient_descent():
    """演示：梯度下降训练"""
    print("\n" + "=" * 50)
    print("4. 梯度下降 + 反向传播")
    print("=" * 50)

    # 训练数据：y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[3], [5], [7], [9], [11]])

    # 初始化参数
    np.random.seed(42)
    W1 = np.random.randn(1, 3) * 0.5
    b1 = np.zeros((1, 3))
    W2 = np.random.randn(3, 1) * 0.5
    b2 = np.zeros((1, 1))

    learning_rate = 0.1

    print(f"训练数据: y = 2x + 1")
    print(f"学习率: {learning_rate}\n")

    # 训练循环
    for epoch in range(1000):
        # 前向传播
        z1 = X @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        y_pred = z2

        # 计算损失
        loss = np.mean((y_pred - y) ** 2)

        # 反向传播
        dz2 = 2 * (y_pred - y) / len(X)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * sigmoid_derivative(z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 更新参数
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}: Loss = {loss:.4f}")

    # 测试
    print(f"\n训练完成！")
    test_x = np.array([[6]])
    z1 = test_x @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    print(f"输入: 6")
    print(f"预测: {z2[0][0]:.4f}")
    print(f"真实: 13 (因为 y = 2×6 + 1)")


# ============================================================
# 第 5 节：用 PyTorch 实现
# ============================================================
def demo_pytorch():
    """演示：用 PyTorch 实现"""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("\n" + "=" * 50)
        print("5. PyTorch 实现")
        print("=" * 50)
        print("需要安装 PyTorch: pip install torch")
        return

    print("\n" + "=" * 50)
    print("5. 用 PyTorch 实现（推荐）")
    print("=" * 50)

    # ============================================================
    # 第一步：定义网络结构
    # ============================================================
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            # nn.Linear(in_features, out_features)：全连接层
            #   - in_features=1：输入特征维度
            #   - out_features=3：输出特征维度（隐藏层有3个神经元）
            self.hidden = nn.Linear(1, 3)  # 隐藏层：1个输入 → 3个神经元
            self.output = nn.Linear(3, 1)  # 输出层：3个输入 → 1个输出

        def forward(self, x):
            # 隐藏层：线性变换 + Sigmoid 激活
            x = torch.sigmoid(self.hidden(x))
            # 输出层：只做线性变换（回归问题不加激活函数）
            x = self.output(x)
            return x

    # ============================================================
    # 第二步：创建模型实例
    # ============================================================
    model = SimpleNet()
    print("模型结构:")
    print(model)

    # 查看参数
    print("\n模型参数:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")

    # ============================================================
    # 第三步：准备训练数据
    # ============================================================
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])

    # ============================================================
    # 第四步：定义损失函数和优化器
    # ============================================================
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam 优化器

    # ============================================================
    # 第五步：训练网络
    # ============================================================
    print("\n开始训练...")
    for epoch in range(1000):
        # 前向传播
        y_pred = model(X)
        loss = criterion(y_pred, y)

        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数

        if epoch % 200 == 0:
            print(f"  Epoch {epoch:4d}: Loss = {loss.item():.4f}")

    # ============================================================
    # 第六步：测试
    # ============================================================
    print("\n训练完成！")
    with torch.no_grad():
        test_x = torch.tensor([[6.0]])
        prediction = model(test_x)
        print(f"输入: 6")
        print(f"预测: {prediction.item():.4f}")
        print(f"真实: 13")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("第一个神经网络\n")
    print("本教程将带你从零实现一个神经网络\n")

    demo_single_neuron()       # 第2节：单个神经元
    demo_two_layer_network()   # 第3节：两层网络
    demo_gradient_descent()    # 第4节：梯度下降
    demo_pytorch()             # 第5节：PyTorch 实现

    print("\n" + "=" * 50)
    print("本节小结")
    print("=" * 50)
    print("1. 前向传播：z = Wx + b, a = σ(z)")
    print("2. 计算损失：L = (1/n)×Σ(ŷ-y)²")
    print("3. 反向传播：∂L/∂w = ∂L/∂ŷ × ∂ŷ/∂w")
    print("4. 更新参数：w = w - η×∂L/∂w")
