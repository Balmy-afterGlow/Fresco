import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict


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


@dataclass
class HyperparameterConfig:
    # === 数据相关超参数 ===
    batch_size: int = 16
    num_workers: int = 4  # 数据加载并行度
    pin_memory: bool = True

    # === 优化器相关超参数 ===
    learning_rate: float = 1e-3  # 稍微提高学习率，加快收敛
    weight_decay: float = 0.01  # L2正则化
    betas: tuple = (0.9, 0.999)  # AdamW的momentum参数
    eps: float = 1e-8  # AdamW的数值稳定性参数

    # === 学习率调度器超参数 ===
    scheduler_type: str = "onecycle"  # onecycle, cosine, step, exponential
    max_lr: float = 1e-3  # OneCycleLR的最大学习率
    pct_start: float = 0.1  # 学习率上升阶段比例
    anneal_strategy: str = "cos"  # 退火策略

    # === 训练相关超参数 ===
    epochs: int = 60  # 增加训练轮数
    warmup_epochs: int = 5  # 预热阶段
    patience: int = 10  # 早停耐心值

    # === 正则化超参数 ===
    dropout: float = 0.3  # Dropout概率
    dropblock_drop_rate: float = 0.1  # DropBlock概率（如果使用）
    label_smoothing: float = 0.1  # 标签平滑

    # === 混合精度训练 ===
    use_amp: bool = True  # 自动混合精度

    # === 指数移动平均 ===
    use_ema: bool = False  # 是否使用EMA
    ema_decay: float = 0.9999  # EMA衰减率

    # === 模型架构超参数 ===
    backbone: str = "resnet50"  # 主干网络
    pretrained: bool = True  # 是否使用预训练权重
    freeze_backbone: bool = False  # 是否冻结主干网络
    freeze_layers: int = 30  # 冻结的层数（从前往后）

    # === 数据增强超参数 ===
    mixup_alpha: float = 0.0  # MixUp参数（设为0禁用）
    cutmix_alpha: float = 0.0  # CutMix参数（设为0禁用）

    # === 损失函数超参数 ===
    loss_function: str = "crossentropy"  # crossentropy, focal, etc.
    focal_alpha: float = 1.0  # Focal Loss的alpha参数
    focal_gamma: float = 2.0  # Focal Loss的gamma参数

    # === 梯度相关超参数 ===
    gradient_clip_val: Optional[float] = 1.0  # 梯度裁剪
    accumulate_grad_batches: int = 1  # 梯度累积


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class ModelEMA:
    """指数移动平均"""

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class LearningRateScheduler:
    """学习率调度器管理"""

    def __init__(self, optimizer, config, steps_per_epoch):
        self.optimizer = optimizer
        self.config = config
        self.steps_per_epoch = steps_per_epoch
        self.scheduler = self._create_scheduler()

    def _create_scheduler(self):
        if self.config.scheduler_type == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.max_lr,
                epochs=self.config.epochs,
                steps_per_epoch=self.steps_per_epoch,
                pct_start=self.config.pct_start,
                anneal_strategy=self.config.anneal_strategy,
            )
        elif self.config.scheduler_type == "cosine":
            return CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.config.epochs // 4, T_mult=2
            )
        else:
            return None

    def step(self, epoch=None):
        if self.scheduler:
            if self.config.scheduler_type == "onecycle":
                self.scheduler.step()
            elif self.config.scheduler_type == "cosine":
                self.scheduler.step(epoch)

    def get_last_lr(self):
        if self.scheduler:
            return self.scheduler.get_last_lr()
        return [self.optimizer.param_groups[0]["lr"]]


class OptimizedTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化组件
        self.criterion = self._create_criterion()
        self.optimizer = self._create_optimizer()
        self.scheduler = LearningRateScheduler(
            self.optimizer, config, len(train_loader)
        )
        self.scaler = GradScaler() if config.use_amp else None
        self.early_stopping = EarlyStopping(patience=config.patience)
        self.ema = (
            ModelEMA(model, decay=config.ema_decay)
            if hasattr(config, "use_ema") and config.use_ema
            else None
        )

        # 训练历史
        self.history = defaultdict(list)
        self.best_val_acc = 0
        self.start_time = time.time()

    def _create_criterion(self):
        """创建损失函数"""
        if self.config.loss_function == "crossentropy":
            return nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        elif self.config.loss_function == "focal":
            # 这里可以实现Focal Loss
            return nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        else:
            return nn.CrossEntropyLoss()

    def _create_optimizer(self):
        """创建优化器"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            eps=self.config.eps,
        )

    def _warmup_lr(self, epoch, warmup_epochs):
        """学习率预热"""
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.config.learning_rate * lr_scale

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 预热学习率
        if epoch < self.config.warmup_epochs:
            self._warmup_lr(epoch, self.config.warmup_epochs)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # 混合精度训练
            if self.config.use_amp and self.scaler:
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()

                # 梯度裁剪
                if self.config.gradient_clip_val:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_val
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # 梯度裁剪
                if self.config.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_val
                    )

                self.optimizer.step()

            # 更新学习率
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()

            # 更新EMA
            if self.ema:
                self.ema.update()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                    "lr": f"{current_lr:.6f}",
                }
            )

        train_loss = running_loss / len(self.train_loader)
        train_acc = 100.0 * correct / total

        return train_loss, train_acc

    def validate(self):
        """验证"""

        if self.ema:
            self.ema.apply_shadow()

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if self.config.use_amp:
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100.0 * correct / total

        if self.ema:
            self.ema.restore()

        return val_loss, val_acc

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.scheduler.state_dict()
            if self.scheduler.scheduler
            else None,
            "val_acc": val_acc,
            # "config": self.config,
            "history": dict(self.history),
        }

        if self.ema:
            checkpoint["ema_state_dict"] = self.ema.shadow
            self.ema.apply_shadow()
            checkpoint["model_state_dict"] = self.model.state_dict()
            self.ema.restore()

        # 保存最新检查点
        torch.save(checkpoint, "latest_checkpoint.pth")

        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, "best_model.pth")
            print(f"💾 Saved best model with validation accuracy: {val_acc:.2f}%")

    def train(self):
        """完整训练流程"""
        print("🚀 Starting training with optimized hyperparameters...")
        print(f"📊 Training on {len(self.train_loader.dataset)} samples")
        print(f"📊 Validating on {len(self.val_loader.dataset)} samples")
        print(f"🔧 Batch size: {self.config.batch_size}")
        print(f"🔧 Learning rate: {self.config.learning_rate}")
        print(f"🔧 Weight decay: {self.config.weight_decay}")
        print(f"🔧 Epochs: {self.config.epochs}")
        print("-" * 60)

        for epoch in range(self.config.epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc = self.validate()

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(self.scheduler.get_last_lr()[0])

            # 打印结果
            elapsed_time = time.time() - self.start_time
            print(f"\nEpoch {epoch + 1}/{self.config.epochs} (⏱️ {elapsed_time:.1f}s):")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")

            # 保存最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc

            self.save_checkpoint(epoch, val_acc, is_best)

            # 早停检查
            if self.early_stopping(val_loss, self.model):
                print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                print(f"🎯 Best validation accuracy: {self.best_val_acc:.2f}%")
                break

        print("\n🎉 Training completed!")
        print(f"🎯 Best validation accuracy: {self.best_val_acc:.2f}%")
        print(
            f"⏱️ Total training time: {(time.time() - self.start_time) / 60:.1f} minutes"
        )

        return dict(self.history)


class AttentionLayer(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.attention(x)
        return x * weights


class OptimizedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        freeze_ratio = 0.7
        total_layers = len(list(self.backbone.named_parameters()))
        freeze_count = int(total_layers * freeze_ratio)
        for i, (name, param) in enumerate(self.backbone.named_parameters()):
            if i < freeze_count:
                param.requires_grad = False
            else:
                param.requires_grad = True

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            AttentionLayer(1024),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
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
