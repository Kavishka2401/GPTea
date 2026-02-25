import json
import io
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# CONFIG 
st.set_page_config(page_title="GPTea - Disease Detection", page_icon="🍃", layout="wide")

MODEL_PATH = "Models/MobileNetV2_TeaDisease_Final.keras"
CLASSES_PATH = "Models/class_names.json"

DISEASE_EXPLANATIONS = {
    "Anthracnose": "Anthracnose typically appears as dark/irregular lesions that may enlarge and merge, causing necrotic patches on leaves.",
    "AlgalLeaf": "Algal leaf spot often shows raised, circular, velvety patches (greenish/orange). Common in humid conditions and older leaves.",
    "RedLeafSpot": "Red leaf spot presents as reddish-brown circular spots that can expand, merge, and reduce photosynthetic leaf area.",
    "WhiteSpot": "White spot appears as pale/whitish lesions, sometimes with darker margins. Lesions can spread and weaken the leaf.",
    "BirdEyeSpot": "Bird’s eye spot shows small round lesions with a pale center and darker ring, resembling an eye-like pattern.",
    "BrownBlight": "Brown blight causes brown necrotic areas often starting at tips/margins and spreading inward in severe cases.",
    "GrayBlight": "Gray blight produces grayish lesions with darker borders; affected areas may look scorched or dried.",
    "Healthy": "Healthy leaves show uniform green color without visible lesions, spots, rust, or abnormal discoloration."
}

# STYLE: header space + footer space 
st.markdown(
    """
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

      /* Whole app background */
      .stApp{ background: var(--bg) !important; }

      /* Remove default page padding (left/right/top/bottom gaps) */
      .block-container{
  padding-top: 0rem !important;
  padding-bottom: 0rem !important;

  padding-left: 2% !important;   /* left space */
  padding-right: 2% !important;  /* right space */

  max-width: 1650px !important;  /* keeps layout centered */
  margin: auto !important;
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

      /* Upload box (MAKE IT WHITE so no green block under title) */
      .subcard{
        background: #ffffff;
        border-radius: var(--radius);
        padding: 16px;
        border: 1px solid rgba(31,90,63,.18);
        margin: 12px 20px 0 20px;
      }

      /* Footer flush */
      .footer-space{
        height: 110px;
        width: 100vw; 
        background: #214f3c;
        margin-left: calc(-50vw + 50%);
      }

      /* File uploader label + helper text bigger */
div[data-testid="stFileUploader"] label p{
  font-size: 20px !important;
  font-weight: 700 !important;
}

/* The small helper lines inside uploader ("No file chosen", "Limit 200MB...") */
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] span{
  font-size: 16px !important;
}

/* "Browse files" button bigger */
div[data-testid="stFileUploader"] button{
  font-size: 18px !important;
  font-weight: 800 !important;
  padding: 0.7rem 1.4rem !important;
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
        div[data-testid="stTextInputRootElement"] { display: none !important; }

    </style>
    <div class="header-space"></div>
    """,
    unsafe_allow_html=True,
)


# MODEL LOAD 
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


# GRAD-CAM HELPERS 
def find_last_conv_layer_name(model: tf.keras.Model) -> str:
    # Try to automatically find the last Conv2D layer in top-level model
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    # If nested model (common in transfer learning), dig into sub-models
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name

    raise ValueError("Could not find a Conv2D layer for Grad-CAM. Set it manually.")


def make_gradcam_heatmap(img_batch: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str, pred_index: int):
    # Use FIRST output if model has multiple outputs
    model_output = model.outputs[0] if isinstance(model.output, (list, tuple)) else model.output

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model_output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_batch)

        # preds could still be list/tuple in some cases
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

    # Resize heatmap to base image size
    heat = tf.image.resize(heatmap[..., np.newaxis], (base_arr.shape[0], base_arr.shape[1])).numpy().squeeze()
    heat = np.clip(heat, 0, 1)

    # Create overlay figure and return as PIL
    fig = plt.figure(figsize=(6, 6), dpi=150)
    plt.axis("off")
    plt.imshow(base_arr)
    plt.imshow(heat, alpha=alpha)  # default colormap
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# PREDICT + XAI (Streamlit-friendly version of your Colab)
def predict_and_explain_xai_pil(
        pil_img: Image.Image,
        model: tf.keras.Model,
        class_names: list,
        DISEASE_INFO: dict,
        threshold: float = 0.60,
        img_size=(224, 224),
):
    # 1) Preprocess (same logic as your Colab, but from PIL instead of file path)
    img = pil_img.convert("RGB").resize(img_size)
    img_array = np.array(img).astype(np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    img_batch = preprocess_input(img_batch)

    # 2) Predict
    raw = model.predict(img_batch, verbose=0)
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    pred = raw[0]
    pred_idx = int(np.argmax(pred))
    pred_class = class_names[pred_idx]
    conf = float(pred[pred_idx])

    # 3) Explanation text (your dictionary)
    explanation = DISEASE_INFO.get(pred_class, "No description available.")

    # 4) Grad-CAM heatmap + overlay
    last_conv_layer_name = find_last_conv_layer_name(model)
    heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer_name, pred_idx)
    overlay_img = overlay_heatmap_on_image(img, heatmap, alpha=0.40)

    low_conf_msg = None
    if conf < threshold:
        low_conf_msg = "Low confidence → image may be unclear / different lighting / not in trained classes."

    # Top-3
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


#  SESSION STATE 
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False

#  BODY UI 
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h1 class="detect-title">Detect Disease</h1>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Upload section
st.markdown('<div class="subcard">', unsafe_allow_html=True)
st.markdown("### Upload image")

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
st.markdown("### Prediction")

st.markdown('<div class="pill">', unsafe_allow_html=True)
predict_btn = st.button(" Detect Disease", use_container_width=True, disabled=not st.session_state.confirmed)
st.markdown("</div>", unsafe_allow_html=True)

if predict_btn:
    try:
        model, class_names = load_model_and_classes()
        pil_img = st.session_state.uploaded_image

        result = predict_and_explain_xai_pil(
            pil_img=pil_img,
            model=model,
            class_names=class_names,
            DISEASE_INFO=DISEASE_EXPLANATIONS,
            threshold=0.60,
            img_size=(224, 224),
        )

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

    except FileNotFoundError as e:
        st.error(
            "Model files not found. Make sure these exist in the same folder as app.py:\n"
            "- MobileNetV2_TeaDisease_Final.keras\n"
            "- class_names.json\n\n"
            f"Error: {e}"
        )
    except Exception as e:
        st.error(f"Something went wrong while predicting: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# Footer green space (blank)
st.markdown('<div class="footer-space"></div>', unsafe_allow_html=True)
