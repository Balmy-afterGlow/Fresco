import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import zipfile
import tempfile
import shutil
import base64
from io import BytesIO
import sys
import json

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_latest import OptimizedCNN

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
        # Try to load from best_model.pth first, then optimized_model.pth
        model_paths = [
            "/home/moyu/Code/Project/Fresco/best_model.pth",
            "/home/moyu/Code/Project/Fresco/train_v3/optimized_model.pth",
        ]

        checkpoint = None
        for model_path in model_paths:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location="cpu")
                print(f"Model loaded from: {model_path}")
                break

        if checkpoint is None:
            raise FileNotFoundError("No model file found")

        model = OptimizedCNN(num_classes=36)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Load class names
        dataset_path = "/home/moyu/Code/Project/Fresco/Dataset/train"
        classes = sorted(os.listdir(dataset_path))

        print("Model loaded successfully!")
        print(f"Number of classes: {len(classes)}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def predict_image(image_path):
    """Predict a single image"""
    global model, classes

    if model is None:
        return None

    # Image preprocessing - same as in predict_latest.py
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probabilities, 5)

        # Convert to lists for JSON serialization
        predictions = []
        for i in range(5):
            predictions.append(
                {
                    "class": classes[top_indices[i]],
                    "probability": float(top_probs[i] * 100),
                }
            )

        return predictions
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        return None


def image_to_base64(image_path):
    """Convert image to base64 for display in browser"""
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            return base64.b64encode(img_data).decode("utf-8")
    except:
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_files():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "No files selected"}), 400

    results = []
    temp_dir = tempfile.mkdtemp()

    try:
        # Process each uploaded file
        for file in files:
            if file and file.filename:
                # Check if it's an image file
                filename = secure_filename(file.filename)
                if not filename.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".gif", ".bmp")
                ):
                    continue

                # Save file temporarily
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)

                # Predict
                predictions = predict_image(file_path)
                if predictions:
                    # Convert image to base64 for display
                    img_base64 = image_to_base64(file_path)
                    results.append(
                        {
                            "filename": filename,
                            "image": img_base64,
                            "predictions": predictions,
                        }
                    )

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route("/upload_zip", methods=["POST"])
def upload_zip():
    if "zipfile" not in request.files:
        return jsonify({"error": "No zip file uploaded"}), 400

    file = request.files["zipfile"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".zip"):
        return jsonify({"error": "Please upload a ZIP file"}), 400

    results = []
    temp_dir = tempfile.mkdtemp()

    try:
        # Save uploaded zip file
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        file.save(zip_path)

        # Extract zip file
        extract_dir = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find all image files in extracted directory
        image_files = []
        for root, dirs, files in os.walk(extract_dir):
            for filename in files:
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                    image_files.append(os.path.join(root, filename))

        # Process each image
        for image_path in image_files:
            predictions = predict_image(image_path)
            if predictions:
                # Get relative filename for display
                rel_filename = os.path.relpath(image_path, extract_dir)
                img_base64 = image_to_base64(image_path)
                results.append(
                    {
                        "filename": rel_filename,
                        "image": img_base64,
                        "predictions": predictions,
                    }
                )

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("Loading model...")
    if load_model():
        print("Starting Flask server...")
        app.run(debug=True, host="0.0.0.0", port=5000)
    else:
        print("Failed to load model. Please check model files.")
