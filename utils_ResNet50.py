import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def explore_data(data_path):
    """Explore and visualize the dataset"""
    print("\nExploring Dataset Structure:")
    print("-" * 50)

    splits = ["train", "validation", "test"]
    for split in splits:
        split_path = os.path.join(data_path, split)
        if os.path.exists(split_path):
            classes = sorted(os.listdir(split_path))
            total_images = sum(
                len(os.listdir(os.path.join(split_path, cls))) for cls in classes
            )

            print(f"\n{split.capitalize()} Set:")
            print(f"Number of classes: {len(classes)}")
            print(f"Total images: {total_images}")
            print(f"Example classes: {', '.join(classes[:5])}...")

    # Visualize sample images
    print("\nVisualizing Sample Images...")
    train_path = os.path.join(data_path, "train")
    classes = sorted(os.listdir(train_path))

    plt.figure(figsize=(15, 10))
    for i in range(9):
        class_name = random.choice(classes)
        class_path = os.path.join(train_path, class_name)
        img_name = random.choice(os.listdir(class_path))
        img_path = os.path.join(class_path, img_name)

        img = Image.open(img_path)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(f"Class: {class_name}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("results/sample_images.png")
    plt.show()


# Visualize augmentations
def show_augmentations(dataset, transform, num_augments=5):
    """Show original image and its augmented versions"""
    idx = random.randint(0, len(dataset) - 1)
    img_path = dataset.images[idx]
    original_img = Image.open(img_path).convert("RGB")

    plt.figure(figsize=(15, 5))

    # Show original
    plt.subplot(1, num_augments + 1, 1)
    plt.imshow(original_img)
    plt.title("Original")
    plt.axis("off")

    # Show augmented versions
    for i in range(num_augments):
        augmented = transform(original_img)
        augmented = augmented.permute(1, 2, 0).numpy()
        augmented = augmented * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        augmented = np.clip(augmented, 0, 1)

        plt.subplot(1, num_augments + 1, i + 2)
        plt.imshow(augmented)
        plt.title(f"Augmented {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("results/augmentations.png")
    plt.show()


# Function to visualize feature maps
# 可视化神经网络内部的特征激活图
def visualize_feature_maps(model, device, sample_image):
    """Visualize feature maps after each conv block"""
    model.eval()  # 确保模型不处于训练状态，关闭dropout等训练专用层

    # Get feature maps after each conv block
    feature_maps = []
    x = sample_image.unsqueeze(0).to(device)

    for block in model.features:
        x = block(x)
        feature_maps.append(
            x.detach().cpu()
        )  # 逐层前向传播，保存每个卷积块的输出特征图（先分离计算图再移回CPU）

    # Plot feature maps
    plt.figure(figsize=(15, 10))
    for i, fmap in enumerate(feature_maps):
        # Plot first 6 channels of each block
        # 取批次中第一个样本的前6个通道
        # 调整维度顺序为HWC（高度、宽度、通道）
        # 归一化到[0,1]范围方便显示
        fmap = fmap[0][:6].permute(1, 2, 0)
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())

        # 为每个卷积块最多显示6个通道
        # 使用viridis色彩映射增强视觉效果
        # 添加适当的标题标识块号和通道号
        for j in range(min(6, fmap.shape[-1])):
            plt.subplot(5, 6, i * 6 + j + 1)
            plt.imshow(fmap[:, :, j], cmap="viridis")
            plt.title(f"Block {i + 1}, Ch {j + 1}")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig("results/feature_maps.png")
    plt.show()


class FruitVegDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv(x)


class FruitVegCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  # 激活BatchNorm和dropout层
    # 累积本轮训练的平均损失和准确率
    running_loss = 0.0
    correct = 0
    total = 0

    # 创建进度条
    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # 梯度清零
        # 前向传播,模型处理图像并计算预测与实际标签之间的损失值
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播,计算损失相对于模型参数的梯度
        loss.backward()
        # 使用计算的梯度更新模型参数
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"}
        )

    return running_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100.0 * correct / total


def plot_training_progress(history):
    """Plot and save training progress"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title("Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/training_progress.png")
    plt.show()


def plot_accuracy_loss(history):
    """Plot training and validation accuracy/loss curves"""
    plt.figure(figsize=(12, 4))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Training", marker="o")
    plt.plot(history["val_acc"], label="Validation", marker="o")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Training", marker="o")
    plt.plot(history["val_loss"], label="Validation", marker="o")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("results/accuracy_loss_curves.png")
    plt.show()

    # Print best accuracy values
    best_train_acc = max(history["train_acc"])
    best_val_acc = max(history["val_acc"])
    print(f"\nBest Training Accuracy: {best_train_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
