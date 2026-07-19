import streamlit as st
import tensorflow as tf
import numpy as np
import time
 
from PIL import Image
 
# ============================================================
# PAGE CONFIGURATION
# ============================================================
 
st.set_page_config(
    page_title="AI Lung Cancer Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ============================================================
# CUSTOM CSS
# ============================================================
 
st.markdown("""
<style>
 
#MainMenu{
visibility:hidden;
}
 
footer{
visibility:hidden;
}
 
header{
visibility:hidden;
}
 
.stApp{
background:linear-gradient(135deg,#0F172A,#1E293B,#111827);
color:white;
}
 
section[data-testid="stSidebar"]{
background:#111827;
}
 
.title{
font-size:50px;
font-weight:800;
color:white;
margin-bottom:0px;
}
 
.subtitle{
font-size:20px;
color:#CBD5E1;
margin-bottom:25px;
}
 
.card{
background:rgba(30,41,59,.82);
backdrop-filter:blur(12px);
padding:25px;
border-radius:22px;
border:1px solid rgba(255,255,255,.08);
box-shadow:0px 15px 35px rgba(0,0,0,.45);
margin-bottom:22px;
transition:.3s;
}
 
.card:hover{
transform:translateY(-3px);
box-shadow:0px 18px 45px rgba(0,0,0,.55);
}
 
.section{
font-size:24px;
font-weight:bold;
color:#38BDF8;
margin-bottom:10px;
}
 
.metric{
background:#1E293B;
padding:18px;
border-radius:18px;
text-align:center;
box-shadow:0px 5px 18px rgba(0,0,0,.25);
}
 
.metric h2{
color:white;
margin-bottom:5px;
}
 
.metric h4{
color:#CBD5E1;
}
 
.predict{
background:#2563EB;
color:white;
border-radius:12px;
padding:10px;
}
 
.stButton>button{
width:100%;
background:linear-gradient(90deg,#2563EB,#3B82F6);
color:white;
font-size:20px;
font-weight:bold;
padding:15px;
border:none;
border-radius:15px;
transition:.3s;
box-shadow:0px 8px 25px rgba(37,99,235,.45);
}
 
.stButton>button:hover{
transform:scale(1.03);
background:linear-gradient(90deg,#1D4ED8,#2563EB);
color:white;
}
 
[data-testid="stFileUploader"]{
background:#1E293B;
padding:20px;
border-radius:15px;
}
 
</style>
""", unsafe_allow_html=True)
 
# ============================================================
# LOAD MODELS
# ============================================================
 
@st.cache_resource
def load_models():
    ct_model = tf.keras.models.load_model("models/ct_model.keras")
    histo_model = tf.keras.models.load_model("models/histo_model.keras")
    return ct_model, histo_model
 
 
ct_model, histo_model = load_models()
 
# ============================================================
# CLASS LABELS
# Order follows flow_from_directory's alphabetical sort of the
# training subfolders — must match training exactly.
# CT_Scan:        Benign cases, Malignant cases, Normal cases
# Histopathology: adenocarcinoma, benign, squamous_cell_carcinoma
# ============================================================
 
CT_CLASSES = ["Benign", "Malignant", "Normal"]
HISTO_CLASSES = ["Adenocarcinoma", "Benign", "Squamous Cell Carcinoma"]
 
# Classes that represent a non-cancerous / negative finding.
# Anything predicted that is NOT in this set is treated as a
# cancer-positive finding for risk-labeling purposes.
NON_CANCEROUS_CLASSES = {"Benign", "Normal"}
 
# ============================================================
# HELPER FUNCTIONS
# ============================================================
 
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image
 
 
def get_risk(predicted_class, confidence):
    is_cancerous_prediction = predicted_class not in NON_CANCEROUS_CLASSES
 
    if is_cancerous_prediction:
        if confidence >= 85:
            return "🔴 High Confidence — Cancerous Finding", "error"
        elif confidence >= 60:
            return "🟠 Moderate Confidence — Cancerous Finding, Recommend Review", "warning"
        else:
            return "🟡 Low Confidence — Inconclusive, Needs Expert Review", "warning"
    else:
        if confidence >= 85:
            return "🟢 High Confidence — Likely Non-Cancerous", "success"
        elif confidence >= 60:
            return "🟡 Moderate Confidence — Likely Non-Cancerous, Recommend Review", "info"
        else:
            return "🟠 Low Confidence — Inconclusive, Needs Expert Review", "warning"
 
 
def predict_image(image, model_type):
    image_array = preprocess_image(image)
 
    start_time = time.time()
 
    if model_type == "CT Scan":
        prediction = ct_model.predict(image_array, verbose=0)
        classes = CT_CLASSES
    else:
        prediction = histo_model.predict(image_array, verbose=0)
        classes = HISTO_CLASSES
 
    end_time = time.time()
 
    probabilities = prediction[0]
    class_index = np.argmax(probabilities)
    predicted_class = classes[class_index]
    confidence = probabilities[class_index] * 100
    prediction_time = end_time - start_time
 
    return predicted_class, confidence, probabilities, classes, prediction_time
 
# ============================================================
# SIDEBAR
# ============================================================
 
st.sidebar.markdown("# 🫁")
 
st.sidebar.title("AI Cancer Detection")
st.sidebar.markdown("---")
 
model_choice = st.sidebar.radio("Choose Dataset", ["CT Scan", "Histopathology"])
 
st.sidebar.markdown("---")
st.sidebar.markdown("### Project Information")
st.sidebar.write("**Model:** ResNet50")
st.sidebar.write("**Framework:** TensorFlow")
 
if model_choice == "CT Scan":
    st.sidebar.write("**Dataset:** IQ-OTH/NCCD")
    sidebar_classes = CT_CLASSES
else:
    st.sidebar.write("**Dataset:** LC25000")
    sidebar_classes = HISTO_CLASSES
 
st.sidebar.write("**Classes:**")
for cls in sidebar_classes:
    st.sidebar.markdown(f"- {cls}")
 
st.sidebar.markdown("---")
st.sidebar.markdown("### Disclaimer")
st.sidebar.caption(
    "This application is developed for educational and research purposes only. "
    "It is not a substitute for professional medical diagnosis."
)
 
# ============================================================
# HEADER
# ============================================================
 
st.markdown('<div class="title">🫁 AI Lung Cancer Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Deep Learning Based Medical Image Classification using ResNet50</div>',
    unsafe_allow_html=True
)
st.markdown("---")
 
# ============================================================
# MAIN LAYOUT
# ============================================================
 
left, right = st.columns([1.4, 1])
 
uploaded_file = None
image = None
 
with left:
    st.markdown('<div class="section">Upload Medical Image</div>', unsafe_allow_html=True)
 
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
 
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width="stretch")
 
with right:
    st.markdown('<div class="section">Prediction Dashboard</div>', unsafe_allow_html=True)
 
    st.info("""
Waiting for prediction...
 
Upload an image
 
Click Predict
 
AI will analyze the image
""")
 
    predict = st.button("🔍 Predict")
 
# ============================================================
# PREDICT BUTTON
# ============================================================
 
if predict:
 
    if uploaded_file is None:
        st.warning("Please upload an image first.")
 
    else:
        with st.spinner("AI is analyzing your image..."):
            (
                predicted_class,
                confidence,
                probabilities,
                class_names,
                prediction_time
            ) = predict_image(image, model_choice)
 
        st.success("Prediction Completed Successfully")
        st.markdown("<br>", unsafe_allow_html=True)
 
        a, b, c = st.columns(3)
 
        with a:
            st.metric("Prediction", predicted_class)
 
        with b:
            st.metric("Confidence", f"{confidence:.2f}%")
 
        with c:
            st.metric("Time", f"{prediction_time:.3f}s")
 
        st.markdown("---")
 
        risk_message, risk_style = get_risk(predicted_class, confidence)
 
        if risk_style == "error":
            st.error(risk_message)
        elif risk_style == "warning":
            st.warning(risk_message)
        elif risk_style == "info":
            st.info(risk_message)
        else:
            st.success(risk_message)
 
        st.caption(
            "This confidence score reflects how certain the model is, not a medical "
            "diagnosis. Scores below ~85% should always be reviewed by a qualified "
            "professional before drawing any conclusion."
        )
 
        st.markdown("## Probability Distribution")
 
        for cls, prob in zip(class_names, probabilities):
            st.write(f"### {cls}")
            st.progress(float(prob))
            st.write(f"{prob * 100:.2f}%")
            st.write("")
 
        st.markdown("---")
        st.subheader("Model Information")
 
        if model_choice == "CT Scan":
            st.info("""
Architecture : ResNet50 (Transfer Learning)
 
Dataset : IQ-OTH/NCCD
 
Classes : 3
 
Image Size : 224 × 224
 
Framework : TensorFlow
            """)
        else:
            st.info("""
Architecture : ResNet50 (Transfer Learning)
 
Dataset : LC25000 Histopathology
 
Classes : 3
 
Image Size : 224 × 224
 
Framework : TensorFlow
            """)
 
        st.markdown("---")
        st.caption(
            "Educational AI Model • Results should not be considered as a medical diagnosis."
        )
 
# ============================================================
# FOOTER
# ============================================================
 
st.markdown("---")
st.caption("AI Lung Cancer Detection System | ResNet50 | TensorFlow | Streamlit")