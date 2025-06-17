#!/bin/bash

# Fresco 项目快速安装脚本

echo "🚀 开始安装 Fresco 水果蔬菜分类器..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 检测到 Python 版本: $python_version"

# 创建虚拟环境（推荐）
echo "📦 创建虚拟环境..."
python3 -m venv fresco_env
source fresco_env/bin/activate

# 升级pip
echo "📦 升级 pip..."
pip install --upgrade pip

# 安装核心依赖
echo "📦 安装项目依赖..."
pip install -r requirements.txt

# 检查CUDA支持
echo "🔍 检查CUDA支持..."
python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA设备数量: {torch.cuda.device_count()}') if torch.cuda.is_available() else print('未检测到CUDA支持')"

echo "✅ 安装完成！"
echo ""
echo "🎯 使用方法："
echo "1. 训练模型: python train_latest.py"
echo "2. 进行预测: python predict_latest.py"  
echo "3. 启动Web应用: cd web && python app.py"
echo "4. 自定义配置: 编辑 utils_latest.py 中的 HyperparameterConfig 类"
echo ""
echo "📊 确保数据集已下载到 Dataset/ 目录"
echo "🔗 数据集下载: https://www.heywhale.com/mw/dataset/676167f5e8187b578b8c17d1/file"
echo ""
echo "💡 提示: 如果使用GPU训练，请确保安装了正确版本的PyTorch"
echo "   访问 https://pytorch.org/ 获取CUDA版本对应的安装命令"
