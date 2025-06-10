# 水果蔬菜分类Web应用

这是一个基于深度学习的水果蔬菜分类Web应用程序，可以批量处理图片并显示分类结果。

## 功能特点

- 📁 支持批量上传多个图片文件
- 📦 支持上传ZIP压缩包批量处理
- 🎯 显示Top 5预测结果和概率
- 📱 响应式设计，支持移动设备
- 🚀 实时处理，无需等待

## 安装和运行

### 1. 安装依赖

```bash
cd /home/moyu/Code/Project/Fresco/web
pip install -r requirements.txt
```

### 2. 确保模型文件存在

确保以下路径之一存在模型文件：
- `/home/moyu/Code/Project/Fresco/best_model.pth`
- `/home/moyu/Code/Project/Fresco/optimized_model.pth`
- `/home/moyu/Code/Project/Fresco/train_v3/optimized_model.pth`

### 3. 启动服务器

```bash
python app.py
```

### 4. 访问应用

打开浏览器访问：http://localhost:5000

## 使用方法

1. **批量上传图片**：
   - 点击"选择图片文件"按钮
   - 选择多个图片文件（支持PNG、JPG、JPEG、GIF、BMP格式）
   - 点击"开始分类"

2. **上传ZIP文件**：
   - 点击"选择ZIP文件"按钮
   - 选择包含图片的ZIP压缩包
   - 点击"开始分类"

## 支持的类别

应用支持识别36种水果和蔬菜：
apple, banana, beetroot, bell pepper, cabbage, capsicum, carrot, cauliflower, chilli pepper, corn, cucumber, eggplant, garlic, ginger, grapes, jalepeno, kiwi, lemon, lettuce, mango, onion, orange, paprika, pear, peas, pineapple, pomegranate, potato, raddish, soy beans, spinach, sweetcorn, sweetpotato, tomato, turnip, watermelon

## 技术栈

- **后端**: Flask + PyTorch
- **前端**: HTML5 + CSS3 + JavaScript
- **模型**: 基于ResNet50的优化CNN
- **图像处理**: PIL + torchvision

## 注意事项

- 单次上传文件大小限制：50MB
- 支持的图片格式：PNG、JPG、JPEG、GIF、BMP
- 建议图片尺寸不要太小，以获得更好的识别效果
