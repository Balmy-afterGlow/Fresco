# 快速使用指南

## 🚀 环境准备

### 1. 克隆项目
```bash
git clone <your-repo-url>
cd Fresco
```

### 2. 安装依赖

#### 方法1: 使用安装脚本（推荐）
```bash
./install.sh
```

#### 方法2: 手动安装
```bash
# 创建虚拟环境
python3 -m venv fresco_env
source fresco_env/bin/activate  # Linux/Mac
# 或 fresco_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 下载数据集
- 访问 [数据集链接](https://www.heywhale.com/mw/dataset/676167f5e8187b578b8c17d1/file)
- 解压到 `Dataset/` 目录，确保结构如下：
```
Dataset/
├── train/
├── validation/
└── test/
```

## 🏃‍♂️ 开始训练

### 基础训练
```bash
python train_latest.py
```

### 自定义配置训练
```python
from utils_latest import *

# 直接修改超参数配置类
config = HyperparameterConfig()
config.epochs = 100              # 增加训练轮数
config.batch_size = 32           # 调整批大小（需要足够GPU内存）
config.learning_rate = 2e-3      # 提高学习率
config.weight_decay = 0.02       # 增加正则化
config.patience = 15             # 调整早停耐心值

# 手动设置训练流程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建数据加载器（参考train_latest.py）
# ... 数据预处理和加载器创建代码 ...

# 创建模型和训练器
model = OptimizedCNN(num_classes=36).to(device)
trainer = OptimizedTrainer(model, train_loader, val_loader, config)
history = trainer.train()
```

## 🔮 模型预测

### 在线预测（URL图像）
```python
from predict_latest import load_model, predict_image

model = load_model()
predict_image("https://example.com/fruit.jpg", model)
```

### 本地图像预测
```python
from PIL import Image
import torch
from utils_latest import OptimizedCNN

# 加载模型
model = OptimizedCNN(num_classes=36)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 预测本地图像
# 修改 predict_latest.py 中的图像加载部分即可
```

## 🌐 Web应用

```bash
cd web
python app.py
```

访问 `http://localhost:5000` 使用Web界面。

## 📊 监控训练

训练过程中会显示：
- 实时损失和准确率
- 学习率变化
- 训练进度条
- 最佳模型保存提示

训练完成后会生成：
- `best_model.pth` - 最佳模型权重
- `latest_checkpoint.pth` - 最新检查点
- `optimized_results.png` - 训练曲线图

## 🛠️ 故障排除

### 常见问题

1. **CUDA内存不足**
   ```python
   config.batch_size = 8  # 减小批大小
   config.num_workers = 2  # 减少工作线程
   ```

2. **训练速度慢**
   ```python
   config.use_amp = True     # 启用混合精度
   config.num_workers = 8    # 增加数据加载线程
   config.pin_memory = True  # 启用内存锁定
   ```

3. **过拟合**
   ```python
   config.weight_decay = 0.02    # 增加正则化
   config.dropout = 0.5          # 增加Dropout
   config.label_smoothing = 0.2  # 增加标签平滑
   ```

4. **欠拟合**
   ```python
   config.learning_rate = 2e-3   # 提高学习率
   config.epochs = 100           # 增加训练轮数
   config.freeze_ratio = 0.5     # 减少冻结层
   ```

### 性能优化建议

1. **数据加载优化**
   - 使用 SSD 存储数据集
   - 调整 `num_workers` 参数
   - 启用 `pin_memory`

2. **训练加速**
   - 启用混合精度训练 (`use_amp=True`)
   - 使用合适的批大小
   - 考虑梯度累积

3. **内存优化**
   - 减小批大小
   - 使用梯度检查点
   - 及时清理无用变量

## 📈 进阶使用

### 自定义数据增强
编辑 `train_latest.py` 中的 `train_transform`：

```python
train_transform = v2.Compose([
    v2.Resize((256, 256)),
    # 添加你的自定义增强
    v2.RandomResizedCrop(224),
    # ... 其他变换
])
```

### 修改模型架构
编辑 `utils_latest.py` 中的 `OptimizedCNN` 类：

```python
class OptimizedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 修改主干网络
        self.backbone = efficientnet_b0(pretrained=True)  # 例如改用EfficientNet
        # ... 其他修改
```

### 实验不同配置
```python
# 直接修改 utils_latest.py 中的 HyperparameterConfig 类
# 或在训练脚本中动态调整

# 尝试不同的学习率调度器
config = HyperparameterConfig()
config.scheduler_type = "cosine"         # 或 "onecycle"
config.learning_rate = 2e-3              # 调整学习率
config.max_lr = 2e-3                     # OneCycleLR的最大学习率

# 尝试不同的正则化策略
config.dropout = 0.5                     # 增加Dropout
config.weight_decay = 0.02               # 增加L2正则化
config.label_smoothing = 0.2             # 增加标签平滑

# 尝试不同的训练策略
config.use_amp = False                   # 关闭混合精度（调试时）
config.batch_size = 8                    # 减小批大小（内存不足时）
config.gradient_clip_val = 0.5           # 调整梯度裁剪
```

## 🎯 最佳实践

1. **开始训练前**
   - 检查数据集完整性
   - 验证GPU可用性
   - 设置合适的随机种子

2. **训练过程中**
   - 监控训练曲线
   - 注意过拟合信号
   - 定期保存检查点

3. **训练完成后**
   - 在测试集上评估模型
   - 分析错误样本
   - 考虑模型集成

4. **部署前**
   - 测试模型推理速度
   - 验证输入输出格式
   - 准备模型版本管理
