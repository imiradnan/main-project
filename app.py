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
# LOGIN + SIGN UP
# ==============================

if "users" not in st.session_state:
    st.session_state.users={}

if "login" not in st.session_state:
    st.session_state.login=False

if "current_user" not in st.session_state:
    st.session_state.current_user=""


if st.session_state.login==False:

    st.title("♻️ Smart Waste AI")

    option=st.radio(
        "Choose Option",
        ["🔐 Login","📝 Sign Up"],
        horizontal=True
    )


    if option=="📝 Sign Up":

        st.subheader("Create Account")

        name=st.text_input("Full Name")

        username=st.text_input("Username")

        password=st.text_input(
            "Password",
            type="password"
        )


        if st.button("Create Account"):

            if username in st.session_state.users:

                st.error("Username already exists")

            else:

                st.session_state.users[username]={
                    "name":name,
                    "password":password
                }

                st.success(
                    "Account Created Successfully 🎉"
                )


    if option=="🔐 Login":

        st.subheader("Login")

        username=st.text_input("Username")

        password=st.text_input(
            "Password",
            type="password"
        )


        if st.button("🚀 Login"):

            if username in st.session_state.users:

                if password==st.session_state.users[username]["password"]:

                    st.session_state.login=True

                    st.session_state.current_user=username

                    st.rerun()

                else:

                    st.error("Wrong Password")

            else:

                st.error("Account Not Found")


    st.stop()



# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_ai_model():

    return tf.keras.models.load_model(
        "smart_waste_model.h5"
    )


model=load_ai_model()



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
"cardboard":"Recycle cardboard properly 📦",
"glass":"Reuse glass items 🍾",
"metal":"Recycle metal waste 🔩",
"paper":"Recycle paper and save trees 📄",
"plastic":"Avoid single use plastic 🥤",
"trash":"Dispose safely 🗑️"
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
        ]

    )


    st.success(
        f"👤 Welcome {st.session_state.current_user}"
    )


    st.success(
        "🟢 System Online"
    )


    if st.button("🚪 Logout"):

        st.session_state.login=False

        st.rerun()



# ==============================
# HOME
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


    st.info(
    """
    Supported Waste:

    📦 Cardboard  
    🍾 Glass  
    🔩 Metal  
    📄 Paper  
    🥤 Plastic  
    🗑 Trash
    """
    )



# ==============================
# CLASSIFY
# ==============================

elif menu=="🤖 Classify Waste":


    st.title("🤖 AI Waste Detection")


    uploaded_file=st.file_uploader(

        "Upload Waste Image",

        type=["jpg","jpeg","png"]

    )


    camera=st.camera_input(
        "Capture Image"
    )


    if camera:

        uploaded_file=camera


    if uploaded_file:


        img=Image.open(uploaded_file).convert("RGB")


        st.image(
            img,
            width=350
        )


        img=img.resize((224,224))


        img_array=image.img_to_array(img)


        img_array=np.expand_dims(
            img_array,
            axis=0
        )


        img_array=preprocess_input(
            img_array
        )


        with st.spinner("AI Checking..."):


            time.sleep(1)


            prediction=model.predict(
                img_array
            )


        index=np.argmax(prediction[0])


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



# ==============================
# DASHBOARD
# ==============================

elif menu=="📊 Dashboard":


    st.title("📊 Dashboard")


    st.metric(
        "AI Accuracy",
        "95%"
    )


    st.metric(
        "Waste Categories",
        "6"
    )


    st.success(
        "AI System Running Successfully 🚀"
    )



# ==============================
# GUIDE
# ==============================

elif menu=="🌱 Recycling Guide":


    st.title("🌱 Recycling Guide")


    st.info(
    """

    📦 Cardboard : Flatten before recycle

    🍾 Glass : Clean before recycle

    🔩 Metal : Separate metals

    📄 Paper : Keep dry

    🥤 Plastic : Reduce usage

    🗑 Trash : Dispose responsibly

    """
    )



# ==============================
# ABOUT
# ==============================

elif menu=="ℹ️ About":


    st.title("ℹ️ About Project")


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

    **Adnan Amin Mir**

    **Obaid Rashid**

    **Nafeesa Jan**

    """
    )


    st.info(
    """

    📞 Customer Support


    Phone:
    +91 7889838588


    Email:
    miradnan227@gmail.com


    Available:

    Monday - Saturday

    Time:

    10:00 AM - 6:00 PM

    """
    )



# ==============================
# FOOTER
# ==============================


st.write("---")


st.markdown(
"""
<div class='footer'>

<h4>♻️ Smart Waste AI System</h4>

</div>
""",
unsafe_allow_html=True
)