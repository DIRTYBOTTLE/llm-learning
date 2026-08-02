"""
第2章 2.4 批归一化
code/04_batch_norm.py

功能：演示 BatchNorm 的效果
对应教程：04_批归一化.md
"""
import torch
import torch.nn as nn


def create_model(use_bn=False):
    """创建是否使用 BatchNorm 的模型"""

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [
                nn.Linear(10, 64),
            ]
            if use_bn:
                layers.append(nn.BatchNorm1d(64))  # 加 BatchNorm
            layers.extend([
                nn.ReLU(),
                nn.Linear(64, 32),
            ])
            if use_bn:
                layers.append(nn.BatchNorm1d(32))  # 加 BatchNorm
            layers.extend([
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ])
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return Net()


def train_model(model, X, y, epochs=200, lr=0.1):
    """训练模型并返回损失历史"""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def main():
    print("=" * 60)
    print("BatchNorm 效果演示")
    print("=" * 60)

    # ===== 生成数据 =====
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100, 1)).float()

    # ===== 对比：有无 BatchNorm =====
    print("\n训练对比（学习率=0.1）：")
    print("-" * 50)

    results = {}
    for use_bn, name in [(False, "无 BatchNorm"), (True, "有 BatchNorm")]:
        torch.manual_seed(42)
        model = create_model(use_bn=use_bn)
        losses = train_model(model, X, y, epochs=200, lr=0.1)

        results[name] = losses
        print(f"{name}: 最终损失 = {losses[-1]:.4f}")

    print("-" * 50)

    # ===== 展示不同学习率的效果 =====
    print("\n不同学习率下的表现：")
    print("-" * 70)
    print(f"{'学习率':<10} {'无 BatchNorm':<15} {'有 BatchNorm':<15} {'BatchNorm 优势':<15}")
    print("-" * 70)

    for lr in [0.01, 0.05, 0.1, 0.5, 1.0]:
        torch.manual_seed(42)
        model_no_bn = create_model(use_bn=False)
        losses_no_bn = train_model(model_no_bn, X, y, epochs=200, lr=lr)

        torch.manual_seed(42)
        model_bn = create_model(use_bn=True)
        losses_bn = train_model(model_bn, X, y, epochs=200, lr=lr)

        diff = losses_no_bn[-1] - losses_bn[-1]
        advantage = f"↓{diff:.3f}" if diff > 0 else f"↑{abs(diff):.3f}"

        print(f"{lr:<10} {losses_no_bn[-1]:<15.4f} {losses_bn[-1]:<15.4f} {advantage:<15}")

    print("-" * 70)

    # ===== BatchNorm 的参数 =====
    print("\nBatchNorm 层的参数：")
    model_bn = create_model(use_bn=True)

    for name, param in model_bn.named_parameters():
        if 'weight' in name and param.numel() > 1:  # 跳过 Linear 的 weight
            continue
        if 'bias' in name and param.numel() > 1:
            continue
        print(f"  {name}: shape={param.shape}")

    # ===== 测试 train/eval 模式差异 =====
    print("\ntrain/eval 模式差异演示：")
    model_bn = create_model(use_bn=True)
    x = torch.randn(5, 10)

    # 训练模式
    model_bn.train()
    y_train1 = model_bn(x)
    y_train2 = model_bn(x)
    print(f"  训练模式两次前向传播结果不同: {not torch.allclose(y_train1, y_train2)}")

    # 测试模式
    model_bn.eval()
    y_test1 = model_bn(x)
    y_test2 = model_bn(x)
    print(f"  测试模式两次前向传播结果相同: {torch.allclose(y_test1, y_test2)}")

    # ===== 总结 =====
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
BatchNorm 的作用：
1. 稳定每层输入分布，加速训练
2. 允许使用更大学习率
3. 减少对初始化的依赖
4. 有轻微的正则化效果

使用位置：通常在 Linear/Conv 后、激活函数前

注意：训练用 model.train()，测试用 model.eval()
""")


if __name__ == '__main__':
    main()
