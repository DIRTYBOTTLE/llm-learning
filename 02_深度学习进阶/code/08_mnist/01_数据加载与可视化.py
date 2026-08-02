"""
第2章 2.8 实战：手写数字识别
08_mnist/01_数据加载与可视化.py

功能：加载 MNIST 数据集，查看数据形状，可视化样本图片
对应教程：08_实战_手写数字识别.md
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def load_data():
    """加载 MNIST 数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),                         # 转为 Tensor，值域 [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))     # MNIST 的均值和标准差
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)

    return train_dataset, test_dataset, train_loader, test_loader


def main():
    print("MNIST 数据加载与可视化\n")

    # 加载数据
    train_dataset, test_dataset, train_loader, test_loader = load_data()

    # 查看数据信息
    print("=" * 50)
    print("MNIST 数据集信息")
    print("=" * 50)
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"训练批次数: {len(train_loader)}")

    # 查看一个 batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch 形状: {images.shape}")   # [64, 1, 28, 28]
    print(f"标签形状: {labels.shape}")       # [64]
    print(f"像素值范围: [{images.min():.3f}, {images.max():.3f}]")

    # 可视化
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle('MNIST 手写数字样本', fontsize=16)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f'标签: {labels[i].item()}', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n已保存图片到 mnist_samples.png")


if __name__ == '__main__':
    main()
