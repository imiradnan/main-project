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
    page_title="Smart Waste Segregation",
    page_icon="♻️",
    layout="wide"
)


# -------------------------
# CSS Design
# -------------------------

st.markdown(
"""
<style>

.title-box {
    background:linear-gradient(90deg,#00b09b,#96c93d);
    padding:25px;
    border-radius:15px;
    text-align:center;
    color:white;
}

.result-card {
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0 5px 20px rgba(0,0,0,0.2);
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
# Login System
# -------------------------

if "login" not in st.session_state:
    st.session_state.login = False


def login_page():

    st.title("🔐 Smart Waste Login")


    username = st.text_input(
        "Username"
    )


    password = st.text_input(
        "Password",
        type="password"
    )


    if st.button("🚀 Login"):

        if username == "adnan" and password == "123@456":

            st.session_state.login = True

            st.success(
                "Login Successful ✅"
            )

            st.rerun()

        else:

            st.error(
                "Invalid Login ❌"
            )



if st.session_state.login == False:

    login_page()

    st.stop()



# -------------------------
# Load Model
# -------------------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "smart_waste_model.h5"
    )


model = load_model()



# -------------------------
# Data
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



tips = {

"cardboard":
"📦 Recycle cardboard boxes properly.",

"glass":
"🍾 Glass can be reused and recycled.",

"metal":
"🔩 Send metals for recycling.",

"paper":
"📄 Save trees by recycling paper.",

"plastic":
"🥤 Reduce plastic usage.",

"trash":
"🗑 Dispose waste correctly."

}



# -------------------------
# Header
# -------------------------

st.markdown(
"""
<div class="title-box">

<h1>♻️ Smart Waste Classification System</h1>

<h4>AI Powered Waste Detection</h4>

</div>
""",
unsafe_allow_html=True
)


st.write("")



# -------------------------
# Sidebar
# -------------------------

with st.sidebar:


    st.title(
        "🌱 Dashboard"
    )


    st.success(
        "Admin Logged In"
    )


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


    if st.button(
        "🚪 Logout"
    ):

        st.session_state.login=False

        st.rerun()



# -------------------------
# Main Section
# -------------------------

col1,col2 = st.columns(2)


uploaded_file = None



with col1:


    st.subheader(
        "📤 Upload Waste Image"
    )


    uploaded_file = st.file_uploader(

        "Choose Image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]

    )


    st.subheader("OR")



    # CAMERA BUTTON FEATURE


    if "camera_open" not in st.session_state:

        st.session_state.camera_open=False



    if st.button(
        "📸 Open Camera"
    ):

        st.session_state.camera_open=True



    if st.session_state.camera_open:


        camera_image = st.camera_input(

            "Capture Waste Image"

        )


        if camera_image:

            uploaded_file=camera_image



        if st.button(
            "❌ Close Camera"
        ):

            st.session_state.camera_open=False

            st.rerun()




# -------------------------
# Prediction
# -------------------------

if uploaded_file is not None:



    img = Image.open(
        uploaded_file
    ).convert("RGB")



    with col1:

        st.image(

            img,

            caption="Selected Image",

            use_container_width=True

        )



    img = img.resize(
        (224,224)
    )


    img_array=image.img_to_array(img)


    img_array=np.expand_dims(
        img_array,
        axis=0
    )


    img_array=img_array/255.0



    with st.spinner(
        "🤖 AI Analyzing Image..."
    ):


        time.sleep(1)


        prediction=model.predict(
            img_array
        )



    index=np.argmax(
        prediction
    )


    result=classes[index]


    confidence=np.max(
        prediction
    )*100





    with col2:


        st.subheader(
            "🤖 Result"
        )


        st.markdown(
        f"""

        <div class="result-card">

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



        if confidence>80:

            st.success(
                "High Confidence ✅"
            )

        elif confidence>50:

            st.warning(
                "Medium Confidence ⚠️"
            )

        else:

            st.error(
                "Low Confidence ❌"
            )



        st.info(
            tips[result]
        )



        report=f"""

SMART WASTE REPORT


Waste Type:

{result}


Confidence:

{confidence:.2f}%


Suggestion:

{tips[result]}

"""


        st.download_button(

            "📥 Download Report",

            report,

            file_name="waste_report.txt"

        )



        if st.button(
            "🔄 Analyze New Image"
        ):

            st.rerun()



else:


    st.info(
        "Upload image or open camera to start"
    )



# -------------------------
# Footer
# -------------------------

st.write("---")


st.markdown(
"""
<div class="footer">

👨‍💻 Developed By  

<h2>
<b>
Adnan Amin Mir <br>
Obaid Rashid <br>
Nafeesa Jan
</b>
</h2>

</div>

""",
unsafe_allow_html=True
)