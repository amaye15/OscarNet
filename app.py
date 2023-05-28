import streamlit as st
from PIL import Image
import io

# Streamlit app
_, title_col, _ = st.columns(3)

with title_col:

    st.title("OscarNet")

_, oscar_col, _ = st.columns(3)
with oscar_col:
    st.image("oscar.png", )


image_data = st.camera_input(label = "Take a photo of some rubbish")

if image_data is not None:
    # Convert the returned bytes array to an image
    image = Image.open(io.BytesIO(image_data.getvalue()))

    #st.image(image, caption='Captured Image.', use_column_width=True)