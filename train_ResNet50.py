import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils_ResNet50 import (
    set_seed,
    explore_data,
    FruitVegDataset,
    FruitVegCNN,
    show_augmentations,
    visualize_feature_maps,
    train_one_epoch,
    validate,
    plot_training_progress,
    plot_accuracy_loss,
)


set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create folder for saving results
os.makedirs("results", exist_ok=True)


# Explore dataset
data_path = "/home/moyu/Code/Project/Fresco/Dataset"
explore_data(data_path)


# Define transforms
train_transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),  # 调整图像大小为224×224像素，适配ResNet50的输入要求
        transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
        transforms.RandomRotation(15),  # 随机旋转图像，角度范围为-15到15度
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        ),  # 随机调整亮度、对比度和饱和度
        transforms.ToTensor(),  # 将PIL图像转换为PyTorch张量(0-1范围)
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),  # 使用ImageNet数据集的均值和标准差进行标准化
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Create datasets and show augmentations
train_dataset = FruitVegDataset(data_path, "train", train_transform)
show_augmentations(train_dataset, train_transform)


# Initialize model and visualize feature maps
model = FruitVegCNN(num_classes=len(train_dataset.classes)).to(device)
sample_image, _ = train_dataset[0]
visualize_feature_maps(model, device, sample_image)


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_dataset = FruitVegDataset(data_path, "validation", val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# Training loop
num_epochs = 30
best_val_acc = 0
history: dict = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print("\nStarting training...")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Update scheduler
    scheduler.step(val_loss)

    # Save history
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Plot progress
    # if (epoch + 1) % 5 == 0:
    #     plot_training_progress(history)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"New best validation accuracy: {best_val_acc:.2f}%")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_val_acc,
            },
            "results/best_model.pth",
        )

# Final training visualization
plot_training_progress(history)


# Plot the curves
plot_accuracy_loss(history)
