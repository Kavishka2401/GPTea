import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
import base64
from groq import Groq

# Page Config
BASE_DIR = Path(__file__).resolve().parent.parent
favicon_path = BASE_DIR / "assets" / "favicon_zoom.ico"

BASE_DIR = Path(__file__).resolve().parent.parent
logo_path = BASE_DIR / "assets" / "logo_white_cropped.png"

# Convert logo to base64
with open(logo_path, "rb") as img_file:
    logo_base64 = base64.b64encode(img_file.read()).decode()

# --- Page Configuration ---
st.set_page_config(page_title="Tea Yield Prediction System",page_icon=str(favicon_path), layout="wide")

# --- Custom CSS for Borders, Colors, and White Background ---
st.markdown("""
    <style>
    /* Full page white background */
    .stApp { background-color: white; }

    /* Main title color */
    h1 { color: #1B4332 !important; }

    /* Feature labels in black */
    label { color: black !important; font-weight: 500; }

    /* Number input styling */
    input[type="number"] {
        background-color: white !important;
        color: black !important;
        border: 1px solid #d3d3d3 !important;
        border-radius: 6px;
        padding: 5px;
        box-shadow: none !important;
    }
    div[data-baseweb="input"] { padding: 0 !important; margin: 0 !important; }

    /* Headings */
    .section-heading {
        color: #1B4332;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 5px;
    }

    /* Columns spacing */
    .section-content { padding-left: 0px; margin-bottom: 15px; }

    /* Selectbox styling */
    .stSelectbox div[data-baseweb="select"] {
        background-color: white !important;
        color: black !important;
        border: 1px solid #d3d3d3 !important;
        border-radius: 6px !important;
        min-height: 38px !important;
    }
    .stSelectbox div[data-baseweb="select"] > div { background-color: white !important; color: black !important; }
    .stSelectbox svg { fill: #1B4332 !important; opacity: 1 !important; }
    .stSelectbox div[data-baseweb="select"]:hover { border: 1px solid #1B4332 !important; }
    
    .header {
        position: sticky;          /* Stays at top when scrolling */
        top: 0;                    /* Sticky offset from top */
        z-index: 999;              /* Sits above other elements */
        background-color: #1B4332; /* Dark green background */
        padding: 10px 10px;        /* Top/Bottom 10px, Left/Right 10px */
        color: white;              /* Default text color inside header */
        display: flex;             /* Flex layout for horizontal alignment */
        justify-content: space-between; /* Space between logo and links */
        align-items: center;       /* Vertically center contents */
        height: 70px;              /* Fixed header height */
        }
    
    .logo {
        display: block;       /* Prevents inline image spacing */
        height: 100%;         /* Fills the height of parent (70px) */
        width: auto;          /* Maintains aspect ratio */
        object-fit: contain;  /* Scales without cropping */
        margin: 0;
        padding: 0;
    }

    .header a {
        color: white;          /* White text for links */
        text-decoration: none; /* No underline */
        margin-left: 20px;     /* Space between links */
        font-weight: bold;     /* Bold text */
    }

    .header a:hover {
        color: #D8F3DC;        /* Light green on hover */
    }
    /* Add padding back to the main content only */
    
    /* Ensure the header stays full width (keep your existing block-container code) */
    .block-container {
        padding: 0rem !important;
        max-width: 100% !important;
    }
    
    /* Footer */
    .footer {
        background-color: #1B4332;
        color: white;
        padding: 20px;
    }
    
    .footer p, .footer a {
        color: white;
    }
    /* Layout helpers */
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
    
    .center-text {
        text-align: center;
    }
    .main-content {
        padding-left: 10rem !important;  /* All sides indented 2rem (adjust as needed) */
        margin: 0 auto;           /* Centers it horizontally */
    }   
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown(f"""
<div class="header">
    <img src="data:image/png;base64,{logo_base64}" class="logo">
    <div style="display:flex; gap:20px;">
        <a href="#home">Home</a>
        <a href="#what">What is GPTea</a>
        <a href="#components">Components</a>
        <a href="#innovation">Research Innovation</a>
        <a href="#team">Research Team</a>
    </div>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    # --- Title ---
    st.title("Tea Yield Prediction System")

    # --- Year and Month ---
    col_y, col_m, col_empty = st.columns([1, 1, 3])
    with col_y:
        year = st.selectbox("Year", options=list(range(2026, 2031)))
    with col_m:
        month = st.selectbox("Month", list(range(1, 13)))

    # --- Weather Data ---
    st.markdown('<div class="section-heading">Weather Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-content">', unsafe_allow_html=True)
    w_col1, w_col2 = st.columns(2)
    with w_col1:
        temperature_2m_mean = st.number_input("Temperature (Mean, °C)", 15.0, 25.0, 19.0, 0.1)
        rain_sum = st.number_input("Rainfall (Total, mm)", 0.0, 1000.0, 100.0, 1.0)
    with w_col2:
        wind_speed_10m_mean = st.number_input("Wind Speed (Mean, m/s)", 0.0, 15.0, 2.0, 0.1)
        et0_fao_evapotranspiration = st.number_input("Evapotranspiration (Mean, mm/day)", 0.0, 10.0, 5.0, 0.1)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Fertilizer Data ---
    st.markdown('<div class="section-heading">Fertilizer Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-content">', unsafe_allow_html=True)
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        T200 = st.number_input("T200 (kg)", 0.0, 1000.0, 0.0, 1.0)
        U709 = st.number_input("U709 (kg)", 0.0, 20000.0, 0.0, 1.0)
        BioShield = st.number_input("BioShield (kg)", 0.0, 500.0, 0.0, 1.0)
    with f_col2:
        Epson = st.number_input("Epson Salt (kg)", 0.0, 500.0, 0.0, 1.0)
        Dolomite = st.number_input("Dolomite (kg)", 0.0, 30000.0, 0.0, 1.0)
        Urea = st.number_input("Urea (kg)", 0.0, 500.0, 0.0, 1.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Lag Data ---
    st.markdown('<div class="section-heading">Lag Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-content">', unsafe_allow_html=True)
    l_col1, l_col2 = st.columns(2)
    with l_col1:
        Compost_lag1 = st.number_input("Last Month Compost (kg)", 0.0, 500.0, 0.0, 1.0)
    with l_col2:
        YPH_lag1 = st.number_input("Last Year YPH", 0.0, 500.0, 0.0, 1.0)
        YPH_lag12 = st.number_input("Last Month YPH", 0.0, 500.0, 0.0, 1.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Load Model ---
    model_path = os.path.join("models", "example_model.pkl")
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
    else:
        st.error(f"Model not found at {model_path}. Please check your path.")

    # --- Predict Button ---
    if st.button("Predict Yield") and os.path.exists(model_path):

        # --- Derived Feature ---
        Dolomite_moisture = Dolomite * (rain_sum - et0_fao_evapotranspiration)

        # --- Create DataFrame (MUST match training feature order) ---
        input_df = pd.DataFrame([[
            T200, U709, Epson, Dolomite, Urea, BioShield,
            temperature_2m_mean, rain_sum, wind_speed_10m_mean, et0_fao_evapotranspiration,
            Dolomite_moisture, YPH_lag1, YPH_lag12, Compost_lag1
        ]], columns=[
            'T200', 'U709', 'Epson Salt', 'Dolomite', 'Urea', 'BioShield',
            'temperature_2m_mean', 'rain_sum', 'wind_speed_10m_mean', 'et0_fao_evapotranspiration',
            'Dolomite_moisture', 'YPH_lag1', 'YPH_lag12', 'Compost_lag1'
        ])

        # --- Predict Yield ---
        predicted_yph = pipeline.predict(input_df)[0]

        # --- Display Predicted YPH ---
        st.markdown(f"""
        <div style="
            color: #1B4332;
            font-weight: bold;
            font-size: 1.6rem;
            margin-top: 15px;
            margin-bottom: 5px;">
            Predicted YPH: {predicted_yph:.2f}
        </div>
        """, unsafe_allow_html=True)

        # ================== AI EXPLANATION USING GROQ ==================
        try:
            # Extract pipeline steps (based on YOUR training code)
            imputer = pipeline.named_steps['imputer']
            scaler = pipeline.named_steps['scaler']
            ridge = pipeline.named_steps['ridge']

            # Apply same preprocessing as training
            X_imputed = imputer.transform(input_df)
            X_scaled = scaler.transform(X_imputed)

            # Calculate feature contributions (Ridge linear model logic)
            contributions = X_scaled[0] * ridge.coef_

            # Create explanation dataframe
            explanation_df = pd.DataFrame({
                "Feature": input_df.columns,
                "Value": input_df.iloc[0].values,
                "Contribution": contributions
            })

            # Sort by importance (optional but smarter)
            explanation_df = explanation_df.sort_values(by="Contribution", key=abs, ascending=False)

            # Build feature summary for AI
            feature_summary = ""
            for _, row in explanation_df.iterrows():
                feature_summary += (
                    f"{row['Feature']} = {row['Value']} "
                    f"(impact on yield: {row['Contribution']:.2f})\n"
                )

            prompt = f"""
    You are an agricultural AI assistant for a Tea Yield Prediction System.
    
    Explain the model prediction in simple, clear, non-technical language for a tea farmer.
    Focus on the most important factors affecting the tea yield.
    
    Feature Contributions:
    {feature_summary}
    """

            # API key
            client = Groq(api_key="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")

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            explanation_text = response.choices[0].message.content

        except Exception as e:
            explanation_text = f"AI explanation unavailable: {str(e)}"

        # --- NON-EDITABLE Explanation Box (WHITE, FIXED, NO TEXT EDITING) ---
        st.markdown(f"""
        <div style="
            color: #1B4332;
            font-weight: bold;
            font-size: 1.3rem;
            margin-top: 10px;
            margin-bottom: 5px;">
            Explanation
        </div>
        <div style="
            background-color: white;
            color: black;
            border: 1px solid #d3d3d3;
            border-radius: 6px;
            padding: 14px;
            font-size: 1rem;
            line-height: 1.7;
            white-space: pre-wrap;">
            {explanation_text}
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
<div class="flex-around">
<div>
<b>GPTea</b><br>AI platform for Sri Lankan farmers
</div>
<div>
<b>Quick Link</b><br>Home | Innovation | Research Team
</div>
<div>
<b>Resources</b><br>Documentation | Research Paper | Dataset
</div>
</div>
<p class="center-text">© 2025 GPTea Research Project. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)

