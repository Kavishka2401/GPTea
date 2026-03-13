import streamlit as st
import base64
from pathlib import Path
from utils.tea_yield_model import load_model, create_input_df, generate_pdf_report, generate_ai_explanation, find_dolomite_moisture

# --- Navigation Helper ---
if st.session_state.get("navigate_to_chat", False):
    st.session_state["navigate_to_chat"] = False
    st.session_state["current_page"] = "04_chatbot"

# --- Page Config ---
BASE_DIR = Path(__file__).resolve().parent.parent
favicon_path = BASE_DIR / "assets" / "favicon_zoom.ico"
logo_path = BASE_DIR / "assets" / "logo_white_cropped.png"

with open(logo_path, "rb") as img_file:
    logo_base64 = base64.b64encode(img_file.read()).decode()

st.set_page_config(page_title="Tea Yield Prediction System",
                   page_icon=str(favicon_path), layout="wide")

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
.stApp { 
background-color: white; 
}

header[data-testid="stHeader"] {
    display: none;
}

/* Main title color */
h1 { 
color: #1B4332 !important; 
}

/* Feature labels in black */
label { 
color: black !important; 
font-weight: 500; 
}

/* Number input styling */
input[type="number"] {
    background-color: white !important;
    color: black !important;
    border: 1px solid #d3d3d3 !important;
    border-radius: 6px;
    padding: 5px;
    box-shadow: none !important;
}

div[data-baseweb="input"] { 
    padding: 0 !important; 
    margin: 0 !important; 
}

/* Headings */
.section-heading {
    color: #1B4332;
    font-weight: bold;
    font-size: 1.2rem;
    margin-bottom: 5px;
}

/* Columns spacing */
.section-content { 
    padding-left: 0px; 
    margin-bottom: 15px; 
}

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

/* Ensure the header stays full width (keep your existing block-container code) */
.block-container {
    padding: 0rem !important;
    max-width: 100% !important;
}

/* Make all buttons same height & full width */
div.stButton > button,
div.stDownloadButton > button {
    background-color: #1B4332;
    color: white;
    border-radius: 5px;
    border: none;
    height: 2.5em;
    width: 100%;
    font-weight: bold;
}

/* Hover effect */
div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background-color: #2D6A4F !important;
    color: white !important;
    border: none;
}

/* Active effect */
div.stButton > button:active,
div.stDownloadButton > button:active {
    background-color: #76c776;
    color: #000000;
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
    
.footer-link {
    text-decoration: none !important; 
    cursor: pointer;
}
    
.footer-link:hover {
    color: Grey;
    text-decoration: none !important; 
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
</style>
""", unsafe_allow_html=True)

# Header Section
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

# --- Main UI ---
col1, col_left_gap, col_main, col_right_gap, col5 = st.columns([0.25, 0.25, 7, 0.25, 0.25])
with col_main:
    st.title("Tea Yield Prediction System")

    # Year & Month
    col_y, col_m, _ = st.columns([1, 1, 3])
    with col_y:
        year = st.selectbox("Year", list(range(2026, 2031)))
    with col_m:
        month = st.selectbox("Month", list(range(1, 13)))

    # Weather Inputs
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

    # Fertilizer Inputs
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

    # Lag Inputs
    st.markdown('<div class="section-heading">Lag Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-content">', unsafe_allow_html=True)
    l_col1, l_col2 = st.columns(2)
    with l_col1:
        Compost_lag1 = st.number_input("Last Month Compost (kg)", 0.0, 500.0, 0.0, 1.0)
    with l_col2:
        YPH_lag12 = st.number_input("Last Year YPH", 0.0, 500.0, 0.0, 1.0)
        YPH_lag1 = st.number_input("Last Month YPH", 0.0, 500.0, 0.0, 1.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # Load model
    pipeline = load_model()
    if pipeline is None:
        st.error("Model not found. Please check your path.")

    if st.button("Predict Yield") and pipeline:
        Dolomite_moisture = find_dolomite_moisture(Dolomite, rain_sum, et0_fao_evapotranspiration )
        input_df = create_input_df(T200, U709, Epson, Dolomite, Urea, BioShield,
                                   temperature_2m_mean, rain_sum, wind_speed_10m_mean, et0_fao_evapotranspiration,
                                   Dolomite_moisture, YPH_lag1, YPH_lag12, Compost_lag1)
        predicted_yph = pipeline.predict(input_df)[0]
        explanation_text = generate_ai_explanation(pipeline, input_df)

        st.session_state.update({
            'predicted_yph': predicted_yph,
            'explanation_text': explanation_text,
            'Dolomite_moisture': Dolomite_moisture
        })

    # Display results if available
    if 'predicted_yph' in st.session_state:
        predicted_yph = st.session_state['predicted_yph']
        explanation_text = st.session_state['explanation_text']
        Dolomite_moisture = st.session_state['Dolomite_moisture']

        st.session_state['last_yield_input'] = {
            "T200": T200,
            "U709": U709,
            "Epson": Epson,
            "Dolomite": Dolomite,
            "Urea": Urea,
            "BioShield": BioShield,
            "temperature_2m_mean": temperature_2m_mean,
            "rain_sum": rain_sum,
            "wind_speed_10m_mean": wind_speed_10m_mean,
            "et0_fao_evapotranspiration": et0_fao_evapotranspiration,
            "Compost_lag1": Compost_lag1,
            "YPH_lag1": YPH_lag1,
            "YPH_lag12": YPH_lag12,
            "predicted_yph": st.session_state['predicted_yph']
        }

        st.markdown(f"""
            <div style="color: #1B4332; font-weight:bold; font-size:1.6rem; margin-top:15px; margin-bottom:5px;">
                Predicted YPH: {predicted_yph:.2f} kg
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""<div style="color:#1B4332; font-weight:bold; font-size:1.3rem; margin-top:10px; margin-bottom:5px;">
                        Explanation</div>
                        <div style="background-color:white; color:black; border:1px solid #d3d3d3; border-radius:6px; padding:14px; font-size:1rem; line-height:1.4;">
                        {explanation_text}</div><br>""", unsafe_allow_html=True)

        pdf_buffer, filename = generate_pdf_report(
            T200, U709, Epson, Dolomite, Urea, BioShield,
            temperature_2m_mean, rain_sum, wind_speed_10m_mean, et0_fao_evapotranspiration,
            Dolomite_moisture, YPH_lag1, YPH_lag12, Compost_lag1,
            predicted_yph, explanation_text, year, month
        )

        # Create two columns for buttons
        btn_col1, btn_col2, _ = st.columns([1, 1.2, 5])

        with btn_col1:
            st.download_button(
                "Download PDF Report",
                pdf_buffer,
                file_name=filename,
                mime="application/pdf",
                key="download_pdf"
            )

        with btn_col2:
            if st.button("Follow to Chatbot"):
                # Store the message for chatbot
                st.session_state['chat_init_message'] = (
                    f"The predicted yield is {st.session_state['predicted_yph']:.2f} kg/ha. "
                    f"Here’s the explanation: {st.session_state['explanation_text']}. "
                    "Please give further recommendations or insights."
                )
                # Redirect to chatbot page
                st.switch_page("pages/04_chatbot.py")

# Footer
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