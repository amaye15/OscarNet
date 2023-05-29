import streamlit as st
from PIL import Image
import io
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, Callback
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator

# Streamlit app
_, title_col, _ = st.columns(3)

with title_col:

    st.title("OscarNet")

_, oscar_col, _ = st.columns(3)
with oscar_col:
    st.image("oscar.png", )


image_data = st.camera_input(label = "Take a photo of some rubbish")

image_size = (224, 224)
class_names = ['battery',
               'biological',
               'brown-glass',
               'cardboard',
               'clothes',
               'green-glass',
               'metal',
               'paper',
               'plastic',
               'shoes',
               'trash',
               'white-glass']

model = Sequential([
    layers.InputLayer(input_shape=(image_size[0], image_size[1], 3)),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.load_weights("model_weights.h5")

if image_data is not None:
    # Convert the returned bytes array to an image
    image = load_img(io.BytesIO(image_data.getvalue()), grayscale=False, target_size=image_size)

    #st.image(image, caption='Captured Image.', use_column_width=True)