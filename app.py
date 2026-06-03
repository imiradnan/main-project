import streamlit as st
import tensorflow as tf
import numpy as np
import time

from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from streamlit_option_menu import option_menu


# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="Smart Waste AI",
    page_icon="♻️",
    layout="wide"
)


# ==============================
# CSS DESIGN
# ==============================

st.markdown(
"""
<style>

.title{
background:linear-gradient(90deg,#00b09b,#96c93d);
padding:30px;
border-radius:20px;
text-align:center;
color:white;
}

.card{
background:white;
padding:25px;
border-radius:20px;
box-shadow:0px 5px 20px rgba(0,0,0,0.2);
text-align:center;
}

.footer{
text-align:center;
color:gray;
}

</style>
""",
unsafe_allow_html=True
)



# ==============================
# LOGIN SYSTEM
# ==============================

if "login" not in st.session_state:
    st.session_state.login=False


if st.session_state.login==False:

    st.title("🔐 Smart Waste Login")

    username=st.text_input("Username")

    password=st.text_input(
        "Password",
        type="password"
    )

    if st.button("🚀 Login"):

        if username=="adnan" and password=="123@456":

            st.session_state.login=True

            st.success("Login Successful")

            st.rerun()

        else:

            st.error("Invalid Login")

    st.stop()



# ==============================
# MODEL LOAD
# ==============================

@st.cache_resource
def load_ai_model():

    return tf.keras.models.load_model(
        "smart_waste_model.h5"
    )


model=load_ai_model()



# ==============================
# CLASSES
# ==============================

classes=[
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


icons={

"cardboard":"📦",
"glass":"🍾",
"metal":"🔩",
"paper":"📄",
"plastic":"🥤",
"trash":"🗑️"

}



tips={

"cardboard":"Recycle cardboard boxes properly 📦",

"glass":"Reuse and recycle glass items 🍾",

"metal":"Send metals for recycling 🔩",

"paper":"Recycle paper and save trees 📄",

"plastic":"Avoid single-use plastic 🥤",

"trash":"Dispose waste safely 🗑️"

}



# ==============================
# SIDEBAR
# ==============================


with st.sidebar:


    menu=option_menu(

        "♻️ Smart Waste AI",

        [
        "🏠 Home",
        "🤖 Classify Waste",
        "📊 Dashboard",
        "🌱 Recycling Guide",
        "ℹ️ About"
        ],

        default_index=0
    )


    st.success("🟢 System Online")


    if st.button("🚪 Logout"):

        st.session_state.login=False

        st.rerun()




# ==============================
# HOME PAGE
# ==============================


if menu=="🏠 Home":


    st.markdown(
    """

    <div class='title'>

    <h1>♻️ Smart Waste Classification System</h1>

    <h3>AI Powered Waste Segregation</h3>

    </div>

    """,
    unsafe_allow_html=True
    )


    st.write("")


    a,b,c=st.columns(3)


    a.metric(
        "Categories",
        "6"
    )


    b.metric(
        "Model",
        "MobileNetV2"
    )


    c.metric(
        "Status",
        "Active ✅"
    )


    st.info(
    """

    This AI system detects waste type from images
    and gives recycling suggestions.

    Supported:

    📦 Cardboard  
    🍾 Glass  
    🔩 Metal  
    📄 Paper  
    🥤 Plastic  
    🗑 Trash  

    """
    )





# ==============================
# CLASSIFICATION PAGE
# ==============================


elif menu=="🤖 Classify Waste":


    st.title(
        "🤖 AI Waste Detection"
    )


    uploaded_file=st.file_uploader(

        "Upload Waste Image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]

    )


    camera=st.camera_input(
        "Or Capture Image"
    )


    if camera:

        uploaded_file=camera



    if uploaded_file:


        img=Image.open(
            uploaded_file
        ).convert("RGB")


        st.image(
            img,
            width=350
        )


        img=img.resize(
            (224,224)
        )


        img_array=image.img_to_array(
            img
        )


        img_array=np.expand_dims(
            img_array,
            axis=0
        )


        img_array=preprocess_input(
            img_array
        )



        with st.spinner(
            "AI Analyzing..."
        ):


            time.sleep(1)


            prediction=model.predict(
                img_array
            )



        index=np.argmax(
            prediction[0]
        )


        result=classes[index]


        confidence=np.max(
            prediction[0]
        )*100



        st.markdown(

        f"""

        <div class='card'>

        <h1>{icons[result]}</h1>

        <h2>{result.upper()}</h2>

        <h3>{confidence:.2f}%</h3>

        </div>

        """,

        unsafe_allow_html=True

        )


        st.progress(
            int(confidence)
        )


        st.success(
            tips[result]
        )



        with st.expander(
            "AI Confidence Details"
        ):

            for i,v in enumerate(prediction[0]):

                st.write(
                    classes[i],
                    round(v*100,2),
                    "%"
                )


        report=f"""

SMART WASTE REPORT

Waste:
{result}

Confidence:
{confidence:.2f}%

Suggestion:
{tips[result]}

"""


        st.download_button(

            "📥 Download Report",

            report,

            "waste_report.txt"

        )




# ==============================
# DASHBOARD PAGE
# ==============================


elif menu=="📊 Dashboard":


    st.title(
        "📊 Waste Dashboard"
    )


    a,b,c=st.columns(3)


    a.metric(
        "Images Tested",
        "100+"
    )

    b.metric(
        "AI Accuracy",
        "95%"
    )

    c.metric(
        "Waste Types",
        "6"
    )


    st.progress(
        95
    )


    st.success(
        "Model Running Successfully 🚀"
    )




# ==============================
# RECYCLING PAGE
# ==============================


elif menu=="🌱 Recycling Guide":


    st.title(
        "🌱 Recycling Guide"
    )


    st.info(
    """

    📦 Cardboard:
    Flatten before recycling


    🍾 Glass:
    Clean before recycling


    🔩 Metal:
    Separate from other waste


    📄 Paper:
    Keep dry


    🥤 Plastic:
    Reduce usage


    🗑 Trash:
    Dispose responsibly

    """
    )




# ==============================
# ABOUT PAGE
# ==============================


elif menu=="ℹ️ About":


    st.title(
        "ℹ️ About Project"
    )


    st.write(
    """

    Smart Waste Segregation System
    using Artificial Intelligence.

    Technologies:

    ✔ Python  
    ✔ TensorFlow  
    ✔ MobileNetV2  
    ✔ Streamlit  

    """
    )


    st.success(
    """

    👨‍💻 Developed By

    Adnan Amin Mir  

    Obaid Rashid  

    Nafeesa Jan

    """
    )




# ==============================
# FOOTER
# ==============================


st.write("---")


st.markdown(
"""

<div class='footer'>

<h4>

♻️ Smart Waste AI System

</h4>

</div>

""",
unsafe_allow_html=True
)