import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import create_df, display_images
from constant import validation_dir

validation_df = create_df(validation_dir)

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

val_images, val_labels = next(validation_generator)

# 加载模型
model = load_model("fresco_model.h5")

# 进行预测
predictions = model.predict(val_images)
pred_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(val_labels, axis=1)

# 获取类别映射
class_indices = validation_generator.class_indices
class_names = {v: k for k, v in class_indices.items()}


# 调用显示函数
display_images(val_images, true_labels, pred_labels, class_names, num_images=9)
