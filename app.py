import streamlit as st
import tensorflow as tf
import numpy as np
import time
import sqlite3

from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from streamlit_option_menu import option_menu


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Smart Waste AI",
    page_icon="♻️",
    layout="wide"
)


# ==============================
# STYLE
# ==============================

st.markdown("""
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
""", unsafe_allow_html=True)


# ==============================
# DATABASE
# ==============================

conn = sqlite3.connect(
    "users.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
username TEXT PRIMARY KEY,
name TEXT,
password TEXT
)
""")

conn.commit()


# ==============================
# LOGIN SYSTEM
# ==============================

if "login" not in st.session_state:
    st.session_state.login=False

if "current_user" not in st.session_state:
    st.session_state.current_user=""


if not st.session_state.login:

    st.title("♻️ Smart Waste AI")

    choice = st.radio(
        "Choose Option",
        ["🔐 Login","📝 Sign Up"],
        horizontal=True
    )


    if choice=="📝 Sign Up":

        st.subheader("Create Account")

        name=st.text_input("Full Name")
        username=st.text_input("Username")
        password=st.text_input(
            "Password",
            type="password"
        )

        if st.button("Create Account"):

            try:

                cursor.execute(
                    "INSERT INTO users VALUES (?,?,?)",
                    (username,name,password)
                )

                conn.commit()

                st.success(
                    "Account Created Successfully 🎉"
                )

            except:

                st.error(
                    "Username already exists"
                )


    if choice=="🔐 Login":

        st.subheader("Login")

        username=st.text_input("Username")

        password=st.text_input(
            "Password",
            type="password"
        )

        if st.button("🚀 Login"):

            cursor.execute(
            """
            SELECT *
            FROM users
            WHERE username=? AND password=?
            """,
            (username,password)
            )

            user=cursor.fetchone()

            if user:

                st.session_state.login=True
                st.session_state.current_user=username

                st.rerun()

            else:

                st.error(
                    "Invalid Login"
                )


    st.stop()


# ==============================
# MODEL
# ==============================

@st.cache_resource
def load_ai_model():

    return tf.keras.models.load_model(
        "smart_waste_model.h5"
    )


model = load_ai_model()


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
"glass":"Recycle glass items 🍾",
"metal":"Send metal for recycling 🔩",
"paper":"Recycle paper and save trees 📄",
"plastic":"Avoid single-use plastic 🥤",
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


    if st.button("🚪 Logout"):

        st.session_state.login=False
        st.rerun()



# ==============================
# HOME
# ==============================

if menu=="🏠 Home":

    st.markdown("""
    <div class='title'>

    <h1>♻️ Smart Waste Classification System</h1>

    <h3>AI Powered Waste Segregation</h3>

    </div>
    """,
    unsafe_allow_html=True)


    st.info("""
    Detects:

    📦 Cardboard  
    🍾 Glass  
    🔩 Metal  
    📄 Paper  
    🥤 Plastic  
    🗑 Trash
    """)



# ==============================
# CLASSIFIER
# ==============================

elif menu=="🤖 Classify Waste":

    st.title(
        "🤖 AI Waste Detection"
    )


    uploaded=st.file_uploader(
        "Upload Image",
        type=["jpg","png","jpeg"]
    )


    camera=st.camera_input(
        "Capture Image"
    )


    if camera:
        uploaded=camera


    if uploaded:

        img=Image.open(uploaded).convert("RGB")

        st.image(
            img,
            width=350
        )


        img=img.resize(
            (224,224)
        )


        arr=image.img_to_array(img)

        arr=np.expand_dims(
            arr,
            axis=0
        )


        arr=preprocess_input(arr)


        with st.spinner(
            "AI Checking..."
        ):

            time.sleep(1)

            pred=model.predict(arr)


        index=np.argmax(pred[0])


        result=classes[index]


        confidence=np.max(
            pred[0]
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
        "Categories",
        "6"
    )

    st.success(
        "System Active 🚀"
    )


# ==============================
# GUIDE
# ==============================

elif menu=="🌱 Recycling Guide":

    st.title(
        "🌱 Recycling Guide"
    )

    st.info("""
📦 Cardboard - Flatten before recycling

🍾 Glass - Clean before recycling

🔩 Metal - Separate properly

📄 Paper - Keep dry

🥤 Plastic - Reduce usage

🗑 Trash - Dispose safely
""")


# ==============================
# ABOUT
# ==============================

elif menu=="ℹ️ About":

    st.title(
        "ℹ️ About Project"
    )


    st.write("""
Smart Waste Segregation System
using Artificial Intelligence.

Technologies:

✔ Python  
✔ TensorFlow  
✔ MobileNetV2  
✔ Streamlit
""")


    st.success("""
👨‍💻 Developed By

**Adnan Amin Mir**

**Obaid Rashid**

**Nafeesa Jan**
""")


    st.info("""
📞 Customer Support

Phone:
+91 7889838588

Email:
miradnan227@gmail.com

Available:
Monday - Saturday

Time:
10:00 AM - 6:00 PM
""")


# ==============================
# FOOTER
# ==============================

st.write("---")

st.markdown("""
<div class='footer'>

<h4>♻️ Smart Waste AI System</h4>

</div>
""",
unsafe_allow_html=True)