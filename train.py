import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from utils import num_of_classes, create_df
from constant import train_dir, test_dir, validation_dir


num_of_classes(train_dir, "train")
num_of_classes(test_dir, "test")
num_of_classes(validation_dir, "validation")


# plot_class_distribution(train_dir)


train_df = create_df(train_dir)
validation_df = create_df(validation_dir)

print(f"训练集总图像数 : {len(train_df)}")
print(f"验证集总图像数 : {len(validation_df)}")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # 对图像进行缩放，将像素值标准化到一个较小的范围
    rotation_range=20,  # 随机旋转图像的角度范围，最多旋转 20 度
    width_shift_range=0.2,  # 水平方向上随机移动图像的比例，最多移动图像宽度的 20%
    height_shift_range=0.2,  # 垂直方向同上
    zoom_range=0.1,  # 随机缩放图像的范围，最多缩放图像 10%
    horizontal_flip=True,  # 随机对图像进行水平翻转
    shear_range=0.1,  # 随机错切变换图像的范围，最多错切图像的 10%
    fill_mode="nearest",  # 对图像进行增强处理时的填充模式，这里设置为最近邻插值
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="file_path",
    y_col="label",
    target_size=(224, 224),  # 指定将图像调整为的目标大小
    color_mode="rgb",  # 指定图像的颜色模式
    class_mode="categorical",  # 指定分类问题的类型
    batch_size=32,
    shuffle=True,  # 指定是否在每个时期之后打乱数据
    seed=42,
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    x_col="file_path",
    y_col="label",
    target_size=(224, 224),
    class_mode="categorical",
    batch_size=32,
    seed=42,
    shuffle=False,
)


# 创建深度卷积神经网络模型
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(36, activation="softmax"),
    ]
)

# 编译模型
# model.compile(
#     optimizer=Adam(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"]
# )

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# 训练模型
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50,
)

# 保存模型
model.save("fresco_model.h5")


plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()
# 部分版本使用的是accuracy
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()
