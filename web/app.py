import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import zipfile
import tempfile
import shutil
import base64
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_v3.utils_ResNet50_enhance import OptimizedCNN

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max file size
app.config["UPLOAD_FOLDER"] = "uploads"

# Create upload folder if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global variable to store the model
model = None
classes = None


def load_model():
    """Load the trained model"""
    global model, classes
    try:
        # 设置模型路径优先级列表
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_paths = [
            os.path.join(base_path, "train_v3", "optimized_model.pth"),
            os.path.join(base_path, "best_model.pth"),
            os.path.join(base_path, "latest_checkpoint.pth"),
        ]

        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError(
                "未找到任何可用的模型文件。请确保以下路径之一存在模型文件:\n"
                + "\n".join(f"  - {path}" for path in model_paths)
            )

        print(f"正在加载模型: {model_path}")
        # 使用map_location='cpu'以确保在没有GPU的环境中也能运行
        checkpoint = torch.load(model_path, map_location="cpu")
        print("模型文件加载完成")

        # 初始化模型，使用与训练时相同的参数
        print("初始化模型架构...")
        model = OptimizedCNN(num_classes=36)

        # 加载模型权重
        print("加载模型权重...")
        model.load_state_dict(checkpoint["model_state_dict"])

        # 将模型设置为评估模式，禁用dropout和batch normalization的训练行为
        print("设置模型为评估模式...")
        model.eval()

        # 尝试使用JIT优化模型
        try:
            # 创建一个示例输入用于追踪
            example_input = torch.rand(1, 3, 224, 224)
            # 使用torch.jit.trace来优化模型
            model = torch.jit.trace(model, example_input)
            print("成功应用JIT优化")
        except Exception as jit_error:
            print(f"JIT优化失败，使用标准模型: {jit_error}")
            # 如果JIT失败，我们仍然可以使用未优化的模型

        # 加载类别名称
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(base_path, "Dataset", "train")

        if os.path.exists(dataset_path):
            classes = sorted(os.listdir(dataset_path))
        else:
            # 如果数据集路径不存在，使用预定义的类别列表
            classes = [
                "apple",
                "banana",
                "beetroot",
                "bell pepper",
                "cabbage",
                "capsicum",
                "carrot",
                "cauliflower",
                "chilli pepper",
                "corn",
                "cucumber",
                "eggplant",
                "garlic",
                "ginger",
                "grapes",
                "jalepeno",
                "kiwi",
                "lemon",
                "lettuce",
                "mango",
                "onion",
                "orange",
                "paprika",
                "pear",
                "peas",
                "pineapple",
                "pomegranate",
                "potato",
                "raddish",
                "soy beans",
                "spinach",
                "sweetcorn",
                "sweetpotato",
                "tomato",
                "turnip",
                "watermelon",
            ]
            print("⚠️ 数据集路径不存在，使用预定义类别列表")

        print("模型加载成功!")
        print(f"类别数量: {len(classes)}")
        return True
    except Exception as e:
        print(f"模型加载错误: {e}")
        return False


def predict_image(image_path):
    """Predict a single image"""
    global model, classes

    if model is None:
        print("错误: 模型未加载")
        return None

    # 定义图像预处理管道 - 与训练时保持一致
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # 调整图像大小
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # 标准化
        ]
    )

    try:
        # 验证文件存在
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return None

        # 验证文件大小
        if os.path.getsize(image_path) > 10 * 1024 * 1024:  # 10MB限制
            print(f"图像文件过大: {image_path}")
            return None

        # 加载并预处理图像
        image = Image.open(image_path).convert("RGB")

        # 添加基本图像验证
        if image.width < 10 or image.height < 10:
            print(f"图像尺寸太小: {image_path}")
            return None

        # 应用图像变换
        input_tensor = transform(image).unsqueeze(0)  # 增加批次维度

        # 进行预测，使用torch.no_grad()提高推理速度并减少内存使用
        with torch.no_grad():
            # 模型推理
            output = model(input_tensor)
            # 转换为概率
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            # 获取前5个预测结果
            top_probs, top_indices = torch.topk(probabilities, 5)

        # 将结果转换为JSON序列化格式
        predictions = []
        for i in range(min(5, len(top_indices))):  # 确保不会超出索引
            idx = top_indices[i].item()  # 转换为Python标量
            if 0 <= idx < len(classes):  # 验证索引范围
                predictions.append(
                    {
                        "class": classes[idx],
                        "probability": float(top_probs[i] * 100),  # 转为百分比
                    }
                )

        return predictions
    except Exception as e:
        print(f"图像预测错误 {image_path}: {e}")
        return None


def image_to_base64(image_path):
    """Convert image to base64 for display in browser"""
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            return base64.b64encode(img_data).decode("utf-8")
    except Exception:
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_files():
    if "files" not in request.files:
        return jsonify({"error": "没有上传文件"}), 400

    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "未选择任何文件"}), 400

    results = []
    temp_dir = tempfile.mkdtemp()

    try:
        valid_files = False
        # Process each uploaded file
        for file in files:
            if file and file.filename:
                # Check if it's an image file
                filename = secure_filename(file.filename)
                if not filename.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")
                ):
                    continue

                valid_files = True

                try:
                    # Save file temporarily
                    file_path = os.path.join(temp_dir, filename)
                    file.save(file_path)

                    # Predict
                    predictions = predict_image(file_path)
                    if predictions:
                        # Convert image to base64 for display
                        img_base64 = image_to_base64(file_path)
                        if img_base64:  # 确保图像正确转换
                            results.append(
                                {
                                    "filename": filename,
                                    "image": img_base64,
                                    "predictions": predictions,
                                }
                            )
                except Exception:
                    # 单个文件处理失败，继续处理其他文件
                    continue

        if not valid_files:
            return jsonify({"error": "未找到支持的图片格式"}), 400

        if not results:
            return jsonify({"error": "无法处理上传的图片"}), 400

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": f"处理图片时出错: {str(e)}"}), 500

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route("/upload_zip", methods=["POST"])
def upload_zip():
    if "zipfile" not in request.files:
        return jsonify({"error": "没有上传ZIP文件"}), 400

    file = request.files["zipfile"]
    if file.filename == "":
        return jsonify({"error": "未选择任何文件"}), 400

    if not file.filename.lower().endswith(".zip"):
        return jsonify({"error": "请上传ZIP格式的文件"}), 400

    results = []
    temp_dir = tempfile.mkdtemp()

    try:
        # Save uploaded zip file
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        file.save(zip_path)

        # Extract zip file
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # 检查压缩包是否有效
                if zip_ref.testzip() is not None:
                    return jsonify({"error": "ZIP文件已损坏"}), 400

                # 安全解压缩
                for member in zip_ref.infolist():
                    # 跳过隐藏文件和目录
                    if member.filename.startswith(
                        "__MACOSX"
                    ) or member.filename.startswith("."):
                        continue
                    # 跳过大于50MB的文件
                    if member.file_size > 50 * 1024 * 1024:
                        continue
                    # 解压缩
                    zip_ref.extract(member, extract_dir)
        except zipfile.BadZipFile:
            return jsonify({"error": "无效的ZIP文件格式"}), 400

        # Find all image files in extracted directory (recursive)
        image_files = []
        for root, dirs, files in os.walk(extract_dir):
            # 跳过隐藏目录
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for filename in files:
                if filename.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")
                ):
                    image_files.append(os.path.join(root, filename))

        if not image_files:
            return jsonify({"error": "ZIP文件中未找到支持的图片格式"}), 400

        # Process each image
        for image_path in image_files:
            try:
                predictions = predict_image(image_path)
                if predictions:
                    # Get relative filename for display
                    rel_filename = os.path.relpath(image_path, extract_dir)
                    img_base64 = image_to_base64(image_path)
                    if img_base64:  # 确保图片转换成功
                        results.append(
                            {
                                "filename": rel_filename,
                                "image": img_base64,
                                "predictions": predictions,
                            }
                        )
            except Exception:
                # 单张图片处理失败时继续处理其他图片
                continue

        if not results:
            return jsonify({"error": "无法处理ZIP文件中的任何图片"}), 400

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": f"处理ZIP文件时出错: {str(e)}"}), 500

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        print("=" * 50)
        print("🍎 水果蔬菜分类应用服务器")
        print("=" * 50)
        print("版本: 1.0.0")
        print("正在加载深度学习模型...")
        if load_model():
            print("模型加载成功!")
            print("-" * 50)
            print("启动 Flask 服务器...")
            print("服务器将在 http://0.0.0.0:5000 上运行")
            print("可通过浏览器访问 http://localhost:5000")
            print("-" * 50)
            app.run(debug=False, host="0.0.0.0", port=5000)
        else:
            print("❌ 模型加载失败。请检查模型文件是否存在。")
            print("请确保以下路径存在模型文件:")
            print("  - /home/moyu/Code/Project/Fresco/train_v3/optimized_model.pth")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 服务器已终止")
    except Exception as e:
        print(f"❌ 启动服务器时出错: {str(e)}")
        sys.exit(1)
