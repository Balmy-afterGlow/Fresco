from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from utils import create_df
from constant import test_dir

test_df = create_df(test_dir)
print(f"测试集总图像数 : {len(test_df)}")

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="file_path",
    y_col="label",
    target_size=(224, 224),
    class_mode="categorical",
    batch_size=32,
    seed=42,
    shuffle=False,
)

# 加载模型
model = load_model("fresco_model.h5")

# 预测和评估
loss, accuracy = model.evaluate(test_generator, steps=50)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
