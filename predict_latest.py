import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from utils_latest import OptimizedCNN


# Load the saved model
def load_model():
    # Check if model file exists
    try:
        # Load model checkpoint
        checkpoint = torch.load("best_model.pth", weights_only=False)
        model = OptimizedCNN(num_classes=36)  # Same as training
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print("Model loaded successfully!")
        return model
    except FileNotFoundError:
        print("Model file 'optimized_model.pth' not found!")
        return None


# Prediction function
def predict_image(url, model):
    # Image preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load image from URL
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    # Transform image
    input_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top 5 predictions
        top_probs, top_indices = torch.topk(probabilities, 5)

    # Show results
    plt.figure(figsize=(12, 4))

    # Show image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    # Show predictions
    plt.subplot(1, 2, 2)
    classes = sorted(os.listdir("/home/moyu/Code/Project/Fresco/Dataset/train"))
    y_pos = range(5)
    plt.barh(y_pos, [prob.item() * 100 for prob in top_probs])
    plt.yticks(y_pos, [classes[idx] for idx in top_indices])
    plt.xlabel("Probability (%)")
    plt.title("Top 5 Predictions")

    plt.tight_layout()
    plt.show()

    # Print predictions
    print("\nPredictions:")
    print("-" * 30)
    for i in range(5):
        print(f"{classes[top_indices[i]]:20s}: {top_probs[i] * 100:.2f}%")


# Load model
model = load_model()

# Now you can use it like this:
predict_image("https://pngimg.com/uploads/watermelon/watermelon_PNG2640.png", model)
