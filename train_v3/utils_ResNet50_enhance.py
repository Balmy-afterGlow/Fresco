import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random


# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


# Improved training configurations
class OptimizedConfig:
    def __init__(self):
        self.image_size = 256  # Increased from 224
        self.batch_size = 16  # Smaller batch size for better generalization
        self.learning_rate = 3e-4
        self.weight_decay = 0.01
        self.epochs = 50
        self.dropout = 0.3


# Optimized model architecture
class OptimizedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Use pretrained ResNet50 as backbone
        self.backbone = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet50", pretrained=True
        )

        # Freeze early layers
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False

        # Modified classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def plot_optimized_results(history):
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(15, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Training", marker="o")
    plt.plot(history["val_acc"], label="Validation", marker="o")
    plt.title("Model Accuracy with Optimizations")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Training", marker="o")
    plt.plot(history["val_loss"], label="Validation", marker="o")
    plt.title("Model Loss with Optimizations")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("optimized_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print best metrics
    best_train_acc = max(history["train_acc"])
    best_val_acc = max(history["val_acc"])
    print(f"\nBest Training Accuracy: {best_train_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
