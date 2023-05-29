import streamlit as st
from PIL import Image
import io
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array

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
    img = load_img(io.BytesIO(image_data.getvalue()), grayscale=False, target_size=image_size)
    # Convert the image to a numpy array and normalize it
    img = img_to_array(img) / 255.0

    # Reshape the image into a batch of size 1
    img = img.reshape(1, image_size[0], image_size[1])

    # Use the model to predict the class of the image
    prediction = model.predict(img)

    # The output is a probability distribution over the 10 classes, so we take the class with the highest probability as the prediction
    predicted_class = prediction.argmax()

    st.title(f"Predicted class: {class_names[predicted_class]}")