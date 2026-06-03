import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image
from PIL import Image


# -------------------------
# Load Model
# -------------------------

model = tf.keras.models.load_model(
    "smart_waste_model.h5"
)


# -------------------------
# Classes
# -------------------------

classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️"
)


st.title(
    "♻️ Smart Waste Classification System"
)


st.write(
    "Upload a waste image and AI will classify it."
)


uploaded_file = st.file_uploader(
    "Choose Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# -------------------------
# Prediction
# -------------------------

if uploaded_file is not None:

    img = Image.open(
        uploaded_file
    ).convert("RGB")


    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )


    img = img.resize(
        (224,224)
    )


    img_array = image.img_to_array(
        img
    )


    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    img_array = img_array / 255.0


    prediction = model.predict(
        img_array
    )


    index = np.argmax(
        prediction
    )


    result = classes[
        index
    ]


    confidence = np.max(
        prediction
    ) * 100


    st.success(
        f"Predicted Category: {result}"
    )


    st.info(
        f"Confidence: {confidence:.2f}%"
    )