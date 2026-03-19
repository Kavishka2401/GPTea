import base64
import json
import io
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from pages.feedback_utils import render_feedback_widget


# Page Config
BASE_DIR = Path(__file__).resolve().parent.parent
favicon_path = BASE_DIR / "assets" / "favicon_zoom.ico"

BASE_DIR = Path(__file__).resolve().parent.parent
logo_path = BASE_DIR / "assets" / "logo_white_cropped.png"

# Convert logo to base64
with open(logo_path, "rb") as img_file:
    logo_base64 = base64.b64encode(img_file.read()).decode()

# CONFIG
st.set_page_config(page_title="GPTea - Disease Detection", page_icon=str(favicon_path), layout="wide")

MODEL_PATH = "Models/MobileNetV2_TeaDisease_Final.keras"
CLASSES_PATH = "Models/class_names.json"
VALIDATOR_PATH = "Models/leaf_validator_best.keras"

DISEASE_EXPLANATIONS = {
    "Anthracnose": "Anthracnose typically appears as dark/irregular lesions that may enlarge and merge, causing necrotic patches on leaves.",
    "AlgalLeaf": "Algal leaf spot often shows raised, circular, velvety patches (greenish/orange). Common in humid conditions and older leaves.",
    "RedLeafSpot": "Red leaf spot presents as reddish-brown circular spots that can expand, merge, and reduce photosynthetic leaf area.",
    "WhiteSpot": "White spot appears as pale/whitish lesions, sometimes with darker margins. Lesions can spread and weaken the leaf.",
    "BirdEyeSpot": "Bird's eye spot shows small round lesions with a pale center and darker ring, resembling an eye-like pattern.",
    "BrownBlight": "Brown blight causes brown necrotic areas often starting at tips/margins and spreading inward in severe cases.",
    "GrayBlight": "Gray blight produces grayish lesions with darker borders; affected areas may look scorched or dried.",
    "Healthy": "Healthy leaves show uniform green color without visible lesions, spots, rust, or abnormal discoloration."
}

# STYLE
st.markdown("""
<style>
:root{
--bg:#ffffff;
--green:#1f5a3f;
--green-dark:#184530;
--light:#dff5b9;
--light-hover:#cfeea0;
--border:rgba(0,0,0,.12);
--radius:18px;
}

header[data-testid="stHeader"] {
    display: none;
}

.header {
    position: sticky;          
    top: 0;                   
    z-index: 999;              
    background-color: #1B4332; 
    padding: 10px 10px;        
    color: white;              
    display: flex;             
    justify-content: space-between; 
    align-items: center;      
    height: 70px;             
}

.logo {
    display: block;       
    height: 100%;         
    width: auto;          
    object-fit: contain; 
    margin: 0;
    padding: 0;
}

.header a {
    color: white;          
    text-decoration: none;
    margin-left: 20px;     
    font-weight: bold;     
}

.header a:hover {
    color: #D8F3DC;     
}

h1 { 
color: #1B4332 !important; 
}

/* Whole app background */
.stApp{ 
    background: var(--bg) !important; 
}

/* Ensure the header stays full width (keep your existing block-container code) */
.block-container {
    padding: 0rem !important;
    max-width: 100% !important;
}

/* Optional: remove extra padding in main area */
section.main > div{
    padding: 0rem !important;
}

/* Header bar flush */
.header-space{
    height: 150px;
    width: 100vw; 
    background: #214f3c;
    border-radius: 0;
    border-bottom: 1px solid var(--border);
    margin-left: calc(-50vw + 50%);
}
/* Thin divider section header */
.card{
    background: transparent;
    padding: 12px 20px;
    border: none;
    border-bottom: 1px solid var(--border);
    margin: 0;
}

/* Make "Detect Disease" title green */
.detect-title{
    color: var(--green) !important;
    font-weight: 800 !important;
    margin: 0 !important;
}

/* 1. The Box: Background and Border */
div[data-testid="stFileUploader"] section {
    background-color: #f2f2f2 !important;
    border: 2px dashed #cfcfcf !important;
    border-radius: 12px !important;
    padding: 25px !important;
}

/* 2. All Text: Targets labels, paragraphs, and spans inside the uploader */
div[data-testid="stFileUploader"] label, 
div[data-testid="stFileUploader"] p, 
div[data-testid="stFileUploader"] span, 
div[data-testid="stFileUploader"] small {
    color: black !important;
}

/* 3. The Button: White background with black text */
div[data-testid="stFileUploader"] button {
    background-color: white !important;
    color: black !important;
    border: 1px solid #cfcfcf !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* 4. Hover effect for the button so it feels interactive */
div[data-testid="stFileUploader"] button:hover {
    background-color: #eeeeee !important;
    border-color: #999999 !important;
}

/* 1. Targets the text inside st.success, st.warning, st.info, etc. */
div[data-testid="stNotificationContent"] {
    color: black !important;
}

/* 2. Targets the filename text that appears after an upload */
div[data-testid="stFileUploaderFileName"] {
    color: black !important;
}

/* 3. Optional: Targets the 'Uploaded' checkmark and progress text */
div[data-testid="stFileUploaderFileData"] {
    color: black !important;
}

/* 4. Targets the 'uploaded' file status container specifically */
[data-testid="stMarkdownContainer"] p {
    color: black !important;
}

/* ---- Buttons (force Streamlit buttons) ---- */
div.stButton > button{
    border-radius: 999px !important;
    background: var(--light) !important;
    color: #0f2f22 !important;
    border: 1px solid rgba(31,90,63,.25) !important;
    font-weight: 1000 !important;
    font-size: 18px !important;
    padding: 1rem 2rem !important;
    }

div.stButton > button:hover{
    background: var(--light-hover) !important;
    border-color: rgba(31,90,63,.40) !important;
    color: #0b2219 !important;
    }

div.stButton > button:active{
    transform: scale(0.99);
    }

div.stButton > button:disabled{
    opacity: .45 !important;
    }

/* Hide that long rounded input box */
div[data-testid="stTextInput"] { display: none !important; }
div[data-testid="stTextInputRootElement"] { 
    display: none !important; 
}

/* Make download button match the green pill buttons */
div.stDownloadButton > button {
    border-radius: 999px !important;
    background: var(--light) !important;
    color: #0f2f22 !important;
    border: 1px solid rgba(31,90,63,.25) !important;
    font-weight: 1000 !important;
    font-size: 18px !important;
    padding: 1rem 2rem !important;
    width: 100% !important;
}
div.stDownloadButton > button:hover {
    background: var(--light-hover) !important;
    border-color: rgba(31,90,63,.40) !important;
    color: #0b2219 !important;
}
div.stDownloadButton > button:active {
    transform: scale(0.99);
}

/* Targets standard st.write text, bold text, and headers */
[data-testid="stMarkdownContainer"] p, 
[data-testid="stMarkdownContainer"] h3, 
[data-testid="stMarkdownContainer"] li {
    color: black !important;
}

/* Targets the 'st.caption' text specifically */
[data-testid="stCaptionContainer"] {
    color: black !important;
}

/* Targets the inline code block (the layer name inside backticks) */
code {
    color: black !important;
    background-color: #e0e0e0 !important;
}

/* Targets st.error messages text */
div[data-testid="stNotificationContent"] {
    color: black !important;
}

.footer {
    background-color: #1B4332;
    color: white;
    padding: 20px;
}

.footer p, .footer a { color: white; }

.footer-link {
    text-decoration: none !important;
    cursor: pointer;
}

.footer-link:hover {
    color: Grey;
    text-decoration: none !important;
}

.flex-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-left: auto;
}

.flex-around {
    display: flex;
    justify-content: space-around;
    align-items: center;
}

.center-text { text-align: center; }  

/* Ensure the footer container and all children are white */
.footer, .footer p, .footer b, .footer span {
    color: #ffffff !important;
}

/* Force all links in the footer to be white and remove underlines */
.footer a, .footer-link {
    color: #ffffff !important;
    text-decoration: none !important;
    transition: opacity 0.3s ease;
}

/* Hover state for links (slight fade instead of color change for a cleaner look) */
.footer a:hover, .footer-link:hover {
    opacity: 0.8;
    color: #ffffff !important;
    text-decoration: underline !important;
}
</style>
""",
unsafe_allow_html=True,)

#HEADER
st.markdown(f"""
<div class="header">
    <img src="data:image/png;base64,{logo_base64}" class="logo">
    <div style="display:flex; gap:10px;">
        <a href="/home" target="_self">Home</a>
        <a href="/tea_yield" target="_self">Yield Prediction Page</a>
        <a href="/disease_detection" target="_self">Disease Detection Page</a>
        <a href="/chatbot" target="_self">Chatbot Page</a>
        <a href="#"></a>
    </div>
</div>
""", unsafe_allow_html=True)

# FEEDBACK
render_feedback_widget(
    page_id   = "disease_detection",
    icon_path = str(BASE_DIR / "assets" / "favicon_zoom.png"),
)

# MODEL LOAD
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


@st.cache_resource
def load_validator_model():
    return tf.keras.models.load_model(VALIDATOR_PATH)


# GRAD-CAM HELPERS
def find_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    raise ValueError("Could not find a Conv2D layer for Grad-CAM. Set it manually.")


def make_gradcam_heatmap(img_batch: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str, pred_index: int):
    model_output = model.outputs[0] if isinstance(model.output, (list, tuple)) else model.output

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model_output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_batch)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = pil_img.convert("RGB")
    base_arr = np.array(base)

    heat = tf.image.resize(heatmap[..., np.newaxis], (base_arr.shape[0], base_arr.shape[1])).numpy().squeeze()
    heat = np.clip(heat, 0, 1)

    fig = plt.figure(figsize=(6, 6), dpi=150)
    plt.axis("off")
    plt.imshow(base_arr)
    plt.imshow(heat, alpha=alpha)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# PREDICT + XAI
def predict_and_explain_xai_pil(
        pil_img: Image.Image,
        model: tf.keras.Model,
        class_names: list,
        DISEASE_INFO: dict,
        threshold: float = 0.60,
        img_size=(224, 224),
):
    img = pil_img.convert("RGB").resize(img_size)
    img_array = np.array(img).astype(np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    img_batch = preprocess_input(img_batch)

    raw = model.predict(img_batch, verbose=0)
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    pred = raw[0]
    pred_idx = int(np.argmax(pred))
    pred_class = class_names[pred_idx]
    conf = float(pred[pred_idx])

    explanation = DISEASE_INFO.get(pred_class, "No description available.")

    last_conv_layer_name = find_last_conv_layer_name(model)
    heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer_name, pred_idx)
    overlay_img = overlay_heatmap_on_image(img, heatmap, alpha=0.40)

    low_conf_msg = None
    if conf < threshold:
        low_conf_msg = "Low confidence → image may be unclear / different lighting / not in trained classes."

    top3_idx = np.argsort(pred)[::-1][:3]
    top3 = [(class_names[int(i)], float(pred[int(i)])) for i in top3_idx]

    return {
        "pred_class": pred_class,
        "confidence": conf,
        "explanation": explanation,
        "heatmap": heatmap,
        "overlay": overlay_img,
        "last_conv_layer": last_conv_layer_name,
        "low_conf_msg": low_conf_msg,
        "top3": top3,
    }


# PDF REPORT
def pil_to_bytes(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return buf


def create_pdf_report(pil_img, result):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["BodyText"]

    story.append(Paragraph("GPTea - Tea Leaf Disease Detection Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Generated On:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Predicted Disease:</b> {result['pred_class']}", normal_style))
    story.append(Paragraph(f"<b>Confidence:</b> {result['confidence']:.2%}", normal_style))
    story.append(Paragraph(f"<b>Explanation:</b> {result['explanation']}", normal_style))

    if result["low_conf_msg"]:
        story.append(Paragraph(f"<b>Warning:</b> {result['low_conf_msg']}", normal_style))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Top 3 Predictions", heading_style))
    for name_i, prob_i in result["top3"]:
        story.append(Paragraph(f"- {name_i}: {prob_i:.2%}", normal_style))

    story.append(Spacer(1, 16))
    story.append(Paragraph("Uploaded Image", heading_style))
    img_bytes = pil_to_bytes(pil_img)
    rl_img = RLImage(img_bytes, width=3 * inch, height=3 * inch)
    story.append(rl_img)

    story.append(Spacer(1, 16))
    story.append(Paragraph("Grad-CAM Overlay", heading_style))
    overlay_bytes = pil_to_bytes(result["overlay"])
    rl_overlay = RLImage(overlay_bytes, width=3 * inch, height=3 * inch)
    story.append(rl_overlay)

    story.append(Spacer(1, 16))
    story.append(Paragraph(f"<b>Grad-CAM Layer Used:</b> {result['last_conv_layer']}", normal_style))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer


# SESSION STATE
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# BODY UI
col1, col_left_gap, col_main, col_right_gap, col5 = st.columns([0.25, 0.25, 7, 0.25, 0.25])
with col_main:
    # --- Title ---
    st.title("Tea Leaf Disease Detection System")

    st.markdown(
        '<h3 style="color: black; margin-bottom: 0.5rem;">Upload image</h3>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Click **Browse files** or drag & drop an image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown('<div class="pill">', unsafe_allow_html=True)
        confirm = st.button(" Confirm Upload", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="pill">', unsafe_allow_html=True)
        reset = st.button("Reset", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if reset:
        st.session_state.uploaded_image = None
        st.session_state.confirmed = False
        st.session_state.prediction_result = None
        st.rerun()

    if uploaded is not None:
        preview_img = st.session_state.uploaded_image = Image.open(io.BytesIO(uploaded.getvalue()))
        preview_img.thumbnail((350, 350))
        st.image(preview_img, caption="Preview")

    if confirm:
        if st.session_state.uploaded_image is None:
            st.warning("Please upload an image first.")
        else:
            st.session_state.confirmed = True
            st.success("Upload confirmed! You can predict now.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Predict section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#1B4332;">Detection</h3>', unsafe_allow_html=True)

    st.markdown('<div class="pill">', unsafe_allow_html=True)
    predict_btn = st.button(" Detect Disease", use_container_width=True, disabled=not st.session_state.confirmed)
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_btn:
        try:
            model, class_names = load_model_and_classes()
            validator = load_validator_model()
            pil_img = st.session_state.uploaded_image

            # ── STEP 1: Leaf Validator ──
            img_resized = pil_img.convert("RGB").resize((224, 224))
            arr = np.array(img_resized).astype(np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)

            val_pred = validator.predict(arr, verbose=0)[0][0]
            leaf_conf = 1 - val_pred  # LEAF_INDEX = 0 → {'Leaf':0, 'NotLeaf':1}
            is_leaf = leaf_conf >= 0.80

            # ── Gate 1: Not a tea leaf → stop ──
            if not is_leaf:
                st.error("Not a tea leaf image. Please upload a valid tea leaf photo.")
                st.write(f"Leaf confidence: **{leaf_conf:.2%}**")
                st.session_state.prediction_result = None

            else:
                # ── STEP 2: Disease Detection ──
                result = predict_and_explain_xai_pil(
                    pil_img=pil_img,
                    model=model,
                    class_names=class_names,
                    DISEASE_INFO=DISEASE_EXPLANATIONS,
                    threshold=0.50,
                    img_size=(224, 224),
                )

                # ── Gate 2: Low confidence → unclear image ──
                if result["confidence"] < 0.40:
                    st.warning("Unclear image — model is not confident enough.")
                    st.write("Please upload a **clearer, well-lit** tea leaf image.")
                    st.write(f"Best guess: **{result['pred_class']}** ({result['confidence']:.2%})")
                    st.session_state.prediction_result = None

                else:
                    # Store result in session state so it persists during reruns
                    st.session_state.prediction_result = result

        except FileNotFoundError as e:
            st.error(
                "Model files not found. Make sure these exist in the same folder as app.py:\n"
                "- MobileNetV2_TeaDisease_Final.keras\n"
                "- leaf_validator_best.keras\n"
                "- class_names.json\n\n"
                f"Error: {e}"
            )
        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")

    # Display Results
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result

        st.success(f"**Disease:** {result['pred_class']}")
        st.write(f"**Confidence:** {result['confidence']:.2%}")
        st.write(f"**Explanation:** {result['explanation']}")

        if result["low_conf_msg"]:
            st.warning(result["low_conf_msg"])

        st.write("**Top 3 predictions:**")
        for name_i, prob_i in result["top3"]:
            st.write(f"- {name_i}: {prob_i:.2%}")

        st.write("### XAI (Grad-CAM)")
        st.caption(f"Grad-CAM last conv layer used: `{result['last_conv_layer']}`")

        c1, c2 = st.columns(2)
        with c1:
            st.image(result["overlay"], caption="Grad-CAM Overlay", width=400)
        with c2:
            st.image(result["heatmap"], caption="Heatmap (raw)", width=400)

        pdf_file = create_pdf_report(st.session_state.uploaded_image, result)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.download_button(
                label="Download PDF Report",
                data=pdf_file,
                file_name="gptea_disease_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with btn_col2:
            if st.button("Follow to Chatbot", use_container_width=True):
                st.session_state['chat_init_message'] = (
                    f"The detected disease is: {result['pred_class']}. "
                    f"Explanation: {result['explanation']}"
                )
                st.switch_page("pages/04_chatbot.py")

            st.markdown("</div>", unsafe_allow_html=True)

# FOOTER
st.markdown("---")
st.markdown("""
<div class="footer">
    <div class="flex-around">
        <div>
            <b><p style="font-size:25px">GPTea</b></p>
           <p style="font-size:18px; max-width: 300px; line-height: 1.4;">
                AI platform for Sri Lankan tea industrials to predict tea yield, 
                detect diseases in leaves and clarify doubts with Chatbot.
           </p>
        </div>
        <div>
            <b><p style="font-size:20px">Quick Links</b><br></p>
            <p>
                <a href="#" class="footer-link">Home</a> | 
                <a href="#" class="footer-link">Innovation</a> | 
                <a href="#" class="footer-link">Research Team</a>
            </p>
        </div>
        <div>
            <b><p>Resources</b><br></p>
            <p>
                <a href="#" class="footer-link">Documentation</a> | 
                <a href="#" class="footer-link">Research Paper</a> | 
                <a href="#" class="footer-link">Dataset</a>
            </p>
        </div>
    </div>
    <br>
    <p class="center-text">© 2026 GPTea Research Project. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)

