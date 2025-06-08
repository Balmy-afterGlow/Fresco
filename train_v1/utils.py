import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def num_of_classes(folder_dir, folder_name):
    classes = [class_name for class_name in os.listdir(folder_dir)]
    print(f"数据集 {folder_name} 中的类别数: {len(classes)}")


def plot_class_distribution(folder_dir):
    classes = [class_name for class_name in os.listdir(folder_dir)]
    count = []
    for class_name in classes:
        count.append(len(os.listdir(os.path.join(folder_dir, class_name))))

    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes, y=count, color="navy")
    plt.xticks(rotation=285)
    for i, value in enumerate(count):
        plt.text(i, value, str(value), ha="center", va="bottom", fontsize=10)
    plt.title("Class Distribution", fontsize=25, fontweight="bold")
    plt.xlabel("Classes", fontsize=15)
    plt.ylabel("Count", fontsize=15)
    plt.yticks(np.arange(0, 105, 10))
    plt.show()


def create_df(folder_dir):
    all_images = []
    classes = [class_name for class_name in os.listdir(folder_dir)]
    for class_name in classes:
        class_path = os.path.join(folder_dir, class_name)
        all_images.extend(
            [
                (os.path.join(class_path, file_name), class_name)
                for file_name in os.listdir(class_path)
            ]
        )
    df = pd.DataFrame(all_images, columns=["file_path", "label"])
    return df


def display_images(images, true_labels, pred_labels, class_names, num_images=9):
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        true_label = class_names[int(true_labels[i])]
        pred_label = class_names[int(pred_labels[i])]
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
