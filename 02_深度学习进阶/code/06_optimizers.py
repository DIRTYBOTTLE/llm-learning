"""
第2章 2.6 优化器详解
code/06_optimizers.py

功能：对比不同优化器的训练效果
对应教程：06_优化器详解.md
"""
import torch
import torch.nn as nn
import torch.optim as optim


def create_model():
    """创建简单网络"""
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )


def train_model(optimizer_name, X, y, epochs=200, lr=0.01):
    """用指定优化器训练模型"""
    torch.manual_seed(42)
    model = create_model()

    # 选择优化器
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "SGD+Momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "AdaGrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    loss_fn = nn.BCELoss()
    losses = []

    for epoch in range(epochs):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def main():
    print("=" * 70)
    print("优化器对比实验")
    print("=" * 70)

    # ===== 生成数据 =====
    torch.manual_seed(42)
    X = torch.randn(200, 10)
    y = (X.sum(dim=1) > 0).float().unsqueeze(1)

    print(f"\n数据：{X.shape[0]} 个样本，{X.shape[1]} 个特征")

    # ===== 对比不同优化器 =====
    optimizers = {
        "SGD": {"lr": 0.01},
        "SGD+Momentum": {"lr": 0.01},
        "AdaGrad": {"lr": 0.05},
        "RMSProp": {"lr": 0.001},
        "Adam": {"lr": 0.001},
        "AdamW": {"lr": 0.001},
    }

    print("\n" + "-" * 70)
    print(f"{'优化器':<15} {'学习率':<10} {'最终损失':<12} {'达到0.3的轮数':<15}")
    print("-" * 70)

    results = {}
    for name, params in optimizers.items():
        losses = train_model(name, X, y, epochs=200, lr=params["lr"])
        results[name] = losses

        # 找到 loss 首次降到 0.3 以下的轮数
        epochs_to_03 = next((i for i, l in enumerate(losses) if l < 0.3), "未达到")

        print(f"{name:<15} {params['lr']:<10} {losses[-1]:<12.4f} {str(epochs_to_03):<15}")

    print("-" * 70)

    # ===== 打印收敛曲线（文本版） =====
    print("\n收敛曲线（每 20 轮打印一次）：")
    print("-" * 70)
    header = f"{'轮数':<8}"
    for name in optimizers.keys():
        header += f"{name:<12}"
    print(header)
    print("-" * 70)

    for epoch in range(0, 200, 20):
        line = f"{epoch:<8}"
        for name in optimizers.keys():
            line += f"{results[name][epoch]:<12.4f}"
        print(line)

    print("-" * 70)

    # ===== 学习率敏感性分析 =====
    print("\nAdam 对学习率的敏感性：")
    print("-" * 50)

    for lr in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        losses = train_model("Adam", X, y, epochs=200, lr=lr)
        print(f"  lr={lr:<8} 最终损失: {losses[-1]:.4f}")

    print("-" * 50)

    # ===== 总结 =====
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
优化器选择指南：

1. SGD + Momentum：
   - 经典选择，泛化可能更好
   - 需要仔细调学习率
   - 适合：计算机视觉任务

2. Adam（推荐）：
   - 自适应学习率，调参简单
   - 收敛快，适合大多数场景
   - 适合：自然语言处理、推荐系统

3. AdamW：
   - Adam + 解耦权重衰减
   - 需要正则化时使用

学习率调度：
- 训练初期：可以用较大学习率
- 训练后期：逐渐降低学习率
- 常用方法：StepLR, CosineAnnealingLR, Warmup
""")


if __name__ == '__main__':
    main()
