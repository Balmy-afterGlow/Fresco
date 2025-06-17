# Fresco - 水果蔬菜分类器
**FR**uit & veg**E**table cla**S**sifi**C**ati**O**n

一个基于深度学习的水果蔬菜图像分类项目，使用优化的 ResNet50 架构实现高精度分类。

## 🎯 项目概述

本项目实现了一个端到端的水果蔬菜图像分类系统，支持 36 种不同类别的水果和蔬菜识别。项目采用了现代化的深度学习技术栈，包括数据增强、混合精度训练、学习率调度等优化策略。

### 📊 数据集信息
- **数据来源**: [工作台 - Heywhale.com](https://www.heywhale.com/mw/dataset/676167f5e8187b578b8c17d1/file)
- **类别数量**: 36 种水果蔬菜
- **数据分割**: 训练集/验证集/测试集
- **支持类别**: 
  - 水果：苹果、香蕉、葡萄、猕猴桃、柠檬、芒果、橙子、梨、菠萝、石榴、西瓜等
  - 蔬菜：甜菜根、甜椒、卷心菜、胡萝卜、花椰菜、辣椒、玉米、黄瓜、茄子、大蒜、生姜等

## 🏗️ 项目结构

```
Fresco/
├── train_latest.py          # 主训练脚本
├── utils_latest.py          # 工具函数和模型定义
├── predict_latest.py        # 预测脚本
├── best_model.pth          # 最佳模型权重
├── latest_checkpoint.pth   # 最新检查点
├── optimized_results.png   # 训练结果可视化
├── Dataset/                # 数据集目录
│   ├── train/              # 训练数据
│   ├── validation/         # 验证数据
│   └── test/               # 测试数据
├── web/                    # Web应用
│   ├── app.py              # Flask应用
│   └── templates/          # 网页模板
└── train_v*/               # 历史版本训练代码
```

## 🔧 技术栈

- **深度学习框架**: PyTorch 2.0+
- **计算机视觉**: torchvision
- **模型架构**: ResNet50 (预训练) + 自定义分类头
- **优化器**: AdamW
- **学习率调度**: OneCycleLR / CosineAnnealingWarmRestarts
- **数据增强**: torchvision.transforms.v2
- **混合精度训练**: torch.amp
- **可视化**: matplotlib
- **Web框架**: Flask

## 🚀 快速开始

### 环境安装

```bash
# 克隆项目
git clone <repository-url>
cd Fresco

# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
python train_latest.py
```

### 进行预测

```bash
python predict_latest.py
```

### 启动Web应用

```bash
cd web
python app.py
```

## 📋 模型训练详细流程

### 1. 数据预处理与增强

#### 训练数据增强策略：
```python
train_transform = v2.Compose([
    v2.Resize((256, 256)),                                    # 图像缩放
    v2.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.7, 1.3)),  # 随机裁剪
    v2.RandomHorizontalFlip(p=0.5),                          # 水平翻转
    v2.RandomVerticalFlip(p=0.4),                            # 垂直翻转
    v2.RandomRotation(degrees=35),                           # 随机旋转
    v2.RandomAffine(degrees=0, translate=(0.2, 0.2), 
                    scale=(0.8, 1.2), shear=(-15, 15, -15, 15)),  # 仿射变换
    v2.RandomPerspective(distortion_scale=0.2, p=0.3),       # 透视变换
    v2.ColorJitter(brightness=0.4, contrast=0.4, 
                   saturation=0.4, hue=0.15),                # 颜色抖动
    v2.RandomAutocontrast(p=0.4),                            # 自动对比度
    v2.RandomEqualize(p=0.3),                                # 直方图均衡化
    v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3),     # 锐度调整
    v2.RandomPosterize(bits=4, p=0.3),                       # 色彩量化
    v2.RandomSolarize(threshold=128, p=0.3),                 # 曝光效果
    v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))], p=0.4),  # 高斯模糊
    v2.RandomErasing(p=0.3, scale=(0.02, 0.25), ratio=(0.3, 3.3)),  # 随机擦除
    v2.ToTensor(),                                           # 转为张量
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准化
])
```

#### 验证数据预处理：
```python
val_transform = v2.Compose([
    v2.Resize((224, 224)),                                   # 固定尺寸
    v2.ToTensor(),                                           # 转为张量
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
])
```

### 2. 模型架构设计

#### 基础架构：
- **主干网络**: ResNet50 (ImageNet预训练权重)
- **冻结策略**: 冻结前70%的层，微调后30%的层
- **分类头**: 多层感知机 + 注意力机制

#### 自定义分类头：
```python
self.backbone.fc = nn.Sequential(
    nn.Linear(num_features, 1024),      # 第一层线性变换
    nn.LayerNorm(1024),                 # 层归一化
    nn.GELU(),                          # GELU激活函数
    nn.Dropout(0.4),                    # Dropout正则化
    AttentionLayer(1024),               # 自注意力层
    nn.Linear(1024, 512),               # 第二层线性变换
    nn.LayerNorm(512),                  # 层归一化
    nn.GELU(),                          # GELU激活函数
    nn.Dropout(0.3),                    # Dropout正则化
    nn.Linear(512, num_classes)         # 输出层
)
```

#### 注意力机制：
```python
class AttentionLayer(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),  # 降维
            nn.ReLU(),                                 # 激活
            nn.Linear(in_features // 4, in_features), # 升维
            nn.Sigmoid()                               # 注意力权重
        )
    
    def forward(self, x):
        weights = self.attention(x)
        return x * weights  # 加权特征
```

### 3. 超参数配置

```python
@dataclass
class HyperparameterConfig:
    # 数据相关
    batch_size: int = 16                    # 批大小
    num_workers: int = 4                    # 数据加载并行度
    pin_memory: bool = True                 # 内存锁定

    # 优化器相关
    learning_rate: float = 1e-3             # 初始学习率
    weight_decay: float = 0.01              # 权重衰减(L2正则化)
    betas: tuple = (0.9, 0.999)             # AdamW动量参数
    eps: float = 1e-8                       # 数值稳定性参数

    # 学习率调度
    scheduler_type: str = "onecycle"        # 调度器类型
    max_lr: float = 1e-3                    # 最大学习率
    pct_start: float = 0.1                  # 学习率上升阶段比例
    anneal_strategy: str = "cos"            # 退火策略

    # 训练相关
    epochs: int = 60                        # 训练轮数
    warmup_epochs: int = 5                  # 预热轮数
    patience: int = 10                      # 早停耐心值

    # 正则化
    dropout: float = 0.3                    # Dropout概率
    label_smoothing: float = 0.1            # 标签平滑

    # 混合精度训练
    use_amp: bool = True                    # 启用自动混合精度

    # 梯度相关
    gradient_clip_val: float = 1.0          # 梯度裁剪
    accumulate_grad_batches: int = 1        # 梯度累积
```

### 4. 训练优化策略

#### 学习率调度策略：
- **OneCycleLR**: 单周期学习率调度，快速收敛
- **预热阶段**: 前5个epoch线性增加学习率
- **退火策略**: 余弦退火降低学习率

#### 正则化技术：
- **权重衰减**: L2正则化防止过拟合
- **Dropout**: 随机失活神经元
- **标签平滑**: 减少过度自信预测
- **梯度裁剪**: 防止梯度爆炸

#### 混合精度训练：
- 使用 `torch.amp.autocast` 和 `GradScaler`
- 加速训练并减少显存占用
- 保持数值稳定性

#### 早停机制：
- 监控验证损失，防止过拟合
- 耐心值为10个epoch
- 自动恢复最佳权重

### 5. 训练流程监控

#### 实时指标追踪：
- 训练/验证 损失和准确率
- 学习率变化
- 训练时间统计

#### 模型保存策略：
- 每个epoch保存最新检查点 (`latest_checkpoint.pth`)
- 验证准确率提升时保存最佳模型 (`best_model.pth`)
- 支持断点续训

#### 可视化输出：
- 训练过程曲线图
- 损失和准确率变化趋势
- 自动保存为 `optimized_results.png`

## 📈 模型性能

项目采用了多种优化技术确保模型性能：

1. **数据增强**: 15+种增强策略提升泛化能力
2. **迁移学习**: 使用ImageNet预训练权重
3. **渐进式解冻**: 逐步解冻网络层进行微调
4. **注意力机制**: 增强特征表示能力
5. **混合精度训练**: 提升训练效率
6. **高级优化器**: AdamW + OneCycleLR组合

## 🔮 使用示例

### 训练新模型
```python
from utils_latest import *
from torch.utils.data import DataLoader
from torchvision.transforms import v2

# 配置超参数（直接修改 HyperparameterConfig 类的默认值）
config = HyperparameterConfig()
config.epochs = 100                      # 增加训练轮数
config.batch_size = 32                   # 调整批大小
config.learning_rate = 2e-3              # 提高学习率
config.weight_decay = 0.02               # 增加正则化
config.use_amp = True                    # 启用混合精度
config.patience = 15                     # 调整早停策略

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建数据集和加载器（参考 train_latest.py 中的完整流程）
# 这里展示核心步骤：
data_path = "/path/to/your/Dataset"
train_dataset = FruitVegDataset(data_path, "train", train_transform)
val_dataset = FruitVegDataset(data_path, "validation", val_transform)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# 创建模型和训练器
model = OptimizedCNN(num_classes=len(train_dataset.classes)).to(device)
trainer = OptimizedTrainer(model, train_loader, val_loader, config)

# 开始训练
history = trainer.train()

# 绘制结果
plot_optimized_results(history)
```

### 进行预测
```python
from utils_latest import OptimizedCNN
import torch

# 加载模型
model = OptimizedCNN(num_classes=36)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 预测图像
predict_image('image_url_or_path', model)
```

## ⚙️ 超参数配置说明

项目的所有超参数都在 `utils_latest.py` 文件中的 `HyperparameterConfig` 类中定义。你可以通过以下方式调整配置：

### 🔧 核心训练参数
```python
config = HyperparameterConfig()

# 基础训练参数
config.epochs = 60                       # 训练轮数
config.batch_size = 16                   # 批大小
config.learning_rate = 1e-3              # 初始学习率
config.weight_decay = 0.01               # 权重衰减(L2正则化)

# 学习率调度
config.scheduler_type = "onecycle"       # 学习率调度器类型
config.max_lr = 1e-3                     # 最大学习率
config.warmup_epochs = 5                 # 预热轮数
config.patience = 10                     # 早停耐心值
```

### 🎯 优化策略参数
```python
# 正则化设置
config.dropout = 0.3                     # Dropout概率
config.label_smoothing = 0.1             # 标签平滑
config.gradient_clip_val = 1.0           # 梯度裁剪

# 训练加速
config.use_amp = True                    # 混合精度训练
config.num_workers = 4                   # 数据加载线程数
config.pin_memory = True                 # 内存锁定

# 模型架构
config.backbone = "resnet50"             # 主干网络
config.pretrained = True                 # 使用预训练权重
```

### 📊 常用配置组合

#### 高性能配置（强力GPU）：
```python
config.epochs = 100
config.batch_size = 32
config.learning_rate = 2e-3
config.num_workers = 8
config.use_amp = True
```

#### 低资源配置（较弱设备）：
```python
config.epochs = 40
config.batch_size = 8
config.learning_rate = 5e-4
config.num_workers = 2
config.use_amp = False
```

#### 调试配置（快速测试）：
```python
config.epochs = 5
config.batch_size = 4
config.patience = 2
config.warmup_epochs = 1
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目！

## 📄 许可证

本项目采用 MIT 许可证。