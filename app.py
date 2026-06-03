import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import time


# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="Smart Waste segration",
    page_icon="♻️",
    layout="wide"
)


# -------------------------
# Custom CSS
# -------------------------

st.markdown(
"""
<style>

.main {
    background-color: #f5f7fa;
}

.title-box {
    background: linear-gradient(90deg,#00b09b,#96c93d);
    padding:25px;
    border-radius:15px;
    text-align:center;
    color:white;
}

.result-card {
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0 4px 15px rgba(0,0,0,0.1);
    text-align:center;
}

.footer {
    text-align:center;
    color:gray;
}

</style>
""",
unsafe_allow_html=True
)


# -------------------------
# Load Model
# -------------------------

@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model(
        "smart_waste_model.h5"
    )


model = load_ai_model()



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


icons = {
    "cardboard":"📦",
    "glass":"🍾",
    "metal":"🔩",
    "paper":"📄",
    "plastic":"🥤",
    "trash":"🗑️"
}



# -------------------------
# Header
# -------------------------

st.markdown(
"""
<div class="title-box">

<h1>♻️ Smart Waste Classification System</h1>

<h4>AI Powered Waste Detection using Deep Learning</h4>

</div>

""",
unsafe_allow_html=True
)


st.write("")



# -------------------------
# Sidebar
# -------------------------

with st.sidebar:

    st.title("🌱 About Project")

    st.write(
    """
    This system uses Artificial Intelligence 
    to classify waste images into different 
    recycling categories.
    """)


    st.info(
    """
    Categories:

    📦 Cardboard  
    🍾 Glass  
    🔩 Metal  
    📄 Paper  
    🥤 Plastic  
    🗑 Trash
    """
    )


    st.write(
        "Developed using TensorFlow + Streamlit"
    )



# -------------------------
# Main Layout
# -------------------------

col1, col2 = st.columns(
    2
)



with col1:

    st.subheader(
        "📤 Upload Waste Image"
    )


    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )



if uploaded_file is not None:


    img = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )


    with col1:

        st.image(
            img,
            caption="Uploaded Image",
            use_container_width=True
        )



    # -------------------------
    # Preprocessing
    # -------------------------


    img_resize = img.resize(
        (224,224)
    )


    img_array = image.img_to_array(
        img_resize
    )


    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    img_array = img_array / 255.0



    with st.spinner(
        "AI is analyzing image..."
    ):

        time.sleep(1)

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



    # -------------------------
    # Result
    # -------------------------


    with col2:


        st.subheader(
            "🤖 AI Result"
        )


        st.markdown(
        f"""

        <div class="result-card">

        <h1>{icons[result]}</h1>

        <h2>{result.upper()}</h2>

        <h3>Confidence</h3>

        <h2>{confidence:.2f}%</h2>


        </div>

        """,
        unsafe_allow_html=True
        )


        st.progress(
            int(confidence)
        )


        if confidence > 80:

            st.success(
                "High confidence prediction ✅"
            )

        elif confidence > 50:

            st.warning(
                "Medium confidence prediction ⚠️"
            )

        else:

            st.error(
                "Low confidence prediction ❌"
            )



else:

    st.info(
        "Upload an image to start classification"
    )



# -------------------------
# Footer
# -------------------------

st.write("---")

st.markdown(
"""
<div class="footer">

Made with ❤️ using Deep Learning | Smart Waste Management System

</div>

""",
unsafe_allow_html=True
)