"""
第2章 2.7 卷积神经网络
code/07_cnn_basics.py

功能：演示卷积操作、池化、CNN 结构
对应教程：07_卷积神经网络.md
"""
import torch
import torch.nn as nn


def demo_convolution():
    """演示卷积操作"""
    print("=" * 60)
    print("1. 卷积操作演示")
    print("=" * 60)

    # 创建一个简单的卷积层
    conv = nn.Conv2d(
        in_channels=1,      # 输入通道（灰度图）
        out_channels=1,     # 输出通道（1个卷积核）
        kernel_size=3,      # 3x3 卷积核
        padding=0,          # 无填充
        bias=False          # 不使用偏置（方便查看权重）
    )

    # 手动设置卷积核（边缘检测）
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]]
        ], dtype=torch.float32).unsqueeze(0))

    print("\n卷积核（Sobel 边缘检测）：")
    print(conv.weight.data.squeeze().numpy())

    # 创建输入图片
    x = torch.tensor([[
        [[1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0]]
    ]], dtype=torch.float32)

    print("\n输入图片 (5x5)：")
    print(x.squeeze().numpy())

    # 卷积
    y = conv(x)
    print("\n卷积输出 (3x3)：")
    print(y.squeeze().detach().numpy())

    # 计算输出尺寸
    print(f"\n输出尺寸计算：")
    print(f"  (5 + 2×0 - 3) / 1 + 1 = {y.shape[2]}x{y.shape[3]}")


def demo_pooling():
    """演示池化操作"""
    print("\n" + "=" * 60)
    print("2. 池化操作演示")
    print("=" * 60)

    x = torch.tensor([[
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]
    ]], dtype=torch.float32)

    print("\n输入 (4x4)：")
    print(x.squeeze().numpy())

    # 最大池化
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    y_max = maxpool(x)
    print("\n最大池化 (2x2, stride=2)：")
    print(y_max.squeeze().numpy())

    # 平均池化
    avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    y_avg = avgpool(x)
    print("\n平均池化 (2x2, stride=2)：")
    print(y_avg.squeeze().numpy())


def demo_padding():
    """演示填充"""
    print("\n" + "=" * 60)
    print("3. 填充演示")
    print("=" * 60)

    x = torch.ones(1, 1, 3, 3)

    print("\n输入 (3x3)：")
    print(x.squeeze().numpy())

    # 无填充
    conv_no_pad = nn.Conv2d(1, 1, 3, padding=0, bias=False)
    y_no_pad = conv_no_pad(x)
    print(f"\n无填充 (padding=0): 输出 {y_no_pad.shape[2]}x{y_no_pad.shape[3]}")

    # 填充 1
    conv_pad1 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    y_pad1 = conv_pad1(x)
    print(f"填充 1 (padding=1):  输出 {y_pad1.shape[2]}x{y_pad1.shape[3]} (尺寸不变!)")


def demo_cnn_structure():
    """演示 CNN 结构"""
    print("\n" + "=" * 60)
    print("4. CNN 结构演示")
    print("=" * 60)

    # 简单 CNN
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = SimpleCNN()
    print("\n模型结构：")
    print(model)

    # 测试输入输出
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    # 各层输出形状
    print("\n各层输出形状变化：")
    x = torch.randn(1, 1, 28, 28)
    for i, layer in enumerate(model.features):
        x = layer(x)
        print(f"  Layer {i} ({layer.__class__.__name__}): {x.shape}")


def demo_parameter_count():
    """对比全连接 vs CNN 参数量"""
    print("\n" + "=" * 60)
    print("5. 参数量对比：全连接 vs CNN")
    print("=" * 60)

    # 全连接网络
    fc_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # CNN
    cnn_model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    fc_params = sum(p.numel() for p in fc_model.parameters())
    cnn_params = sum(p.numel() for p in cnn_model.parameters())

    print(f"\n全连接网络参数量: {fc_params:,}")
    print(f"CNN 参数量:       {cnn_params:,}")
    print(f"CNN 节省了:       {fc_params - cnn_params:,} 参数 ({(fc_params-cnn_params)/fc_params:.1%})")


def main():
    demo_convolution()
    demo_pooling()
    demo_padding()
    demo_cnn_structure()
    demo_parameter_count()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
CNN 的核心思想：
1. 局连接：每个神经元只连接一小块区域
2. 权值共享：所有位置用同一个卷积核
3. 池化：下采样，减少计算量

输出尺寸公式：
  H_out = (H_in + 2P - K) / S + 1

经典架构：
- LeNet-5 (1998)：手写数字识别
- AlexNet (2012)：ImageNet 冠军
- ResNet (2015)：残差连接，解决深度网络训练
""")


if __name__ == '__main__':
    main()
