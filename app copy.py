import streamlit as st
from PIL import Image
import io

# Streamlit app
st.title("Image Classification App")

image_data = st.camera_input(label = "Take a photo of some rubbish")

if image_data is not None:
    # Convert the returned bytes array to an image
    image = Image.open(io.BytesIO(image_data.getvalue()))

    st.image(image, caption='Captured Image.', use_column_width=True)