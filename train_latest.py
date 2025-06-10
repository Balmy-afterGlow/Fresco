import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from utils_latest import (
    set_seed,
    FruitVegDataset,
    HyperparameterConfig,
    OptimizedCNN,
    OptimizedTrainer,
    plot_optimized_results,
)


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # 配置超参数
    config = HyperparameterConfig()

    # 数据路径
    data_path = "/home/moyu/Code/Project/Fresco/Dataset"

    # 获取数据变换
    train_transform = v2.Compose(
        [
            v2.Resize((256, 256)),
            v2.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.7, 1.3)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.4),
            v2.RandomRotation(degrees=35),
            v2.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                scale=(0.8, 1.2),
                shear=(-15, 15, -15, 15),
            ),
            v2.RandomPerspective(distortion_scale=0.2, p=0.3),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            v2.RandomAutocontrast(p=0.4),
            v2.RandomEqualize(p=0.3),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            v2.RandomPosterize(bits=4, p=0.3),
            v2.RandomSolarize(threshold=128, p=0.3),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))], p=0.4),
            v2.RandomErasing(p=0.3, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
            v2.ToTensor(),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToTensor(),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # 创建数据集
    train_dataset = FruitVegDataset(data_path, "train", train_transform)
    val_dataset = FruitVegDataset(data_path, "validation", val_transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # 创建模型
    model = OptimizedCNN(num_classes=len(train_dataset.classes)).to(device)

    # 创建训练器并开始训练
    trainer = OptimizedTrainer(model, train_loader, val_loader, config)
    history = trainer.train()

    # 绘制结果
    plot_optimized_results(history)


if __name__ == "__main__":
    main()
