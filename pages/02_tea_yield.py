import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
import base64
from groq import Groq
from fpdf import FPDF
from io import BytesIO

# Page Config
BASE_DIR = Path(__file__).resolve().parent.parent
favicon_path = BASE_DIR / "assets" / "favicon_zoom.ico"

BASE_DIR = Path(__file__).resolve().parent.parent
logo_path = BASE_DIR / "assets" / "logo_white_cropped.png"

# Convert logo to base64
with open(logo_path, "rb") as img_file:
    logo_base64 = base64.b64encode(img_file.read()).decode()

def generate_pdf_report(
    T200, U709, Epson, Dolomite, Urea, BioShield,
    temperature_2m_mean, rain_sum, wind_speed_10m_mean, et0_fao_evapotranspiration,
    Dolomite_moisture, YPH_lag1, YPH_lag12, Compost_lag1,
    predicted_yph, explanation_text, year=None, month=None
):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, f"Tea Yield Report - {month}/{year}" if month and year else "Tea Yield Report", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, f"Predicted YPH: {predicted_yph:.2f}", ln=True)
    pdf.ln(5)

    pdf.multi_cell(0, 10, "Explanation:\n" + explanation_text)
    pdf.ln(5)

    pdf.multi_cell(0, 10, "Input Variables:\n" +
                   f"T200: {T200}, U709: {U709}, Epson: {Epson}, Dolomite: {Dolomite}, Urea: {Urea}, BioShield: {BioShield}\n" +
                   f"Temperature: {temperature_2m_mean}, Rainfall: {rain_sum}, Wind speed: {wind_speed_10m_mean}, Evapotranspiration: {et0_fao_evapotranspiration}\n" +
                   f"Dolomite Moisture: {Dolomite_moisture}, YPH_lag1: {YPH_lag1}, YPH_lag12: {YPH_lag12}, Compost_lag1: {Compost_lag1}"
                  )

    # Use BytesIO with dest='S' (return string) then encode
    pdf_bytes = pdf.output(dest='S').encode('latin1')

    pdf_buffer = BytesIO(pdf_bytes)
    pdf_buffer.seek(0)

    filename = f"Tea_Yield_Report_{month}_{year}.pdf" if month and year else "Tea_Yield_Report.pdf"
    return pdf_buffer, filename

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

/* Target the button by its label text or general button style */
div.stButton > button {
    background-color: #1B4332;
    color: white;
    border-radius: 5px;
    border: none;
    height: 2em;
    width: 100%;
    font-weight: bold;
}

/* Hover effect */
div.stButton > button:hover {
    background-color: #2D6A4F !important;
    color: white !important;
    border: none;
}

div.stDownloadButton > button {
    background-color: #1B4332;
    color: white;
    border-radius: 8px;
    border: none;
    height: 2.5em;
    width: 100%;
    font-weight: bold;
}

/* Hover effect */
div.stDownloadButton > button:hover {
    background-color: #2D6A4F !important;
    color: white !important;
    border: none;
}
   
/* Active/Click effect */
div.stButton > button:active {
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
        <a href="#">Home</a>
        <a href="#"></a>
        <a href="#"></a>
        <a href="#"></a>
        <a href="#"></a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Main Content with Column Indentation ---
col1, col_left_gap, col_main, col_right_gap, col5 = st.columns([0.25, 0.25, 7, 0.25, 0.25])
with col_main:
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
    model_path = os.path.join("models", "Final_Ridge_Model_23_24_25.pkl")
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
    else:
        st.error(f"Model not found at {model_path}. Please check your path.")

    # -------------------- PREDICT BUTTON --------------------
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

            # Sort by importance
            explanation_df = explanation_df.sort_values(by="Contribution", key=abs, ascending=False)

            # Build feature summary for AI
            feature_summary = ""
            for _, row in explanation_df.iterrows():
                feature_summary += (
                    f"{row['Feature']} = {row['Value']} "
                    f"(impact on yield: {row['Contribution']:.2f})\n"
                )

            prompt = f"""
                You are GPTea AI, an agricultural assistant that explains tea yield predictions to farmers in simple, easy-to-understand English. 
            Your explanation must include: Do not copy the example output. The following is for reference only indentation and everything must be same. Generate explanation only for the input below.

            IMPORTANT NOTES TO CONSIDER
            1. Keep explanations tight. Do not add extra blank lines. Display each bullet consecutively without extra spacing. Leave spacing like how its mentioned in the example input.
            2. Do not copy the example output. The following is for reference only. Generate explanation only for the input below.

            1. Introduction of yourself.
            2. Past yield information (last month and same month last year).
            3. Weather variables impact.
            4. Fertilizer variables impact (including T200, U709, Epson, Dolomite, Urea, BioShield, Compost_lag1).
            5. Recommendations to improve yield.
            6. Do NOT repeat the same sentence exactly; make the sentences varied.
            7. Use **human-friendly names** for features:
               - temperature_2m_mean → Temperature
               - rain_sum → Rainfall
               - wind_speed_10m_mean → Wind speed
               - et0_fao_evapotranspiration → Evapotranspiration
               - Dolomite_moisture → Dolomite Moisture
               - Other features use the same name
            8. Avoid technical jargon; be simple and conversational.
            9. Explain the contribution of each variable in a **positive or negative way**.

            Do not copy the example output. The following is for reference only. Generate explanation only for the input below.
            ---
            Below is a example output for a given input.
            # Example Input:
            Predicted Yield: 340
            Past Yield Last Month: 320
            Past Yield Same Month Last Year: 300

            Feature Contributions:
            Temperature = 28.5 (impact: +0.40)
            Rainfall = 150 (impact: +1.30)
            Wind speed = 3.2 (impact: -0.25)
            Evapotranspiration = 5.8 (impact: +0.10)
            Dolomite Moisture = 16 (impact: -0.50)
            T200 = 6 (impact: +0.75)
            U709 = 3 (impact: +0.50)
            Epson = 2 (impact: +0.20)
            Dolomite = 8 (impact: -0.60)
            Urea = 5 (impact: +0.35)
            BioShield = 1 (impact: -0.10)
            Compost_lag1 = 20 (impact: +0.50)

            # Example Output:
            Hi, I am GPTea AI. I am going to explain why I predicted this yield.
            In the same month of last year, the yield was 300 and last month it was 320, which explains part of the current prediction.\n            
            Weather Variables:
            1. The temperature is 28.5, which slightly increased the yield.
            2. The rainfall was 150 mm, contributing strongly to a higher yield.
            3. Wind speed of 3.2 slightly reduced the yield.
            4. Evapotranspiration of 5.8 had a small positive effect.
            5. Dolomite Moisture of 16 slightly decreased the yield.\n
            Fertilizer Variables:
            1. T200 fertilizer significantly boosted the yield.
            2. U709 also contributed positively.
            3. Epson had a small positive impact.
            4. Dolomite slightly lowered the yield.
            5. Urea improved the yield moderately.
            6. BioShield had a minor negative effect.
            7. Compost_lag1 helped improve the yield slightly.\n
            Recommendations:
            - Consider optimizing the Dolomite and BioShield application to minimize negative impact.
            - Maintain proper irrigation and monitor rainfall patterns.
            - Keep applying beneficial fertilizers like T200, U709, Urea, and Compost regularly to improve yield further.
            - Ensure temperature and wind conditions are favorable during growth periods.
            Do not copy the example output. The following is for reference only. Generate explanation only for the input below.
            ---
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

        # --- Save to Session State ---
        st.session_state['predicted_yph'] = predicted_yph
        st.session_state['explanation_text'] = explanation_text
        st.session_state['Dolomite_moisture'] = Dolomite_moisture

    # -------------------- DISPLAY RESULTS --------------------
    if 'predicted_yph' in st.session_state:
        predicted_yph = st.session_state['predicted_yph']
        explanation_text = st.session_state['explanation_text']
        Dolomite_moisture = st.session_state['Dolomite_moisture']

        # --- Display Prediction ---
        st.markdown(f"""
        <div style="color: #1B4332; font-weight:bold; font-size:1.6rem; margin-top:15px; margin-bottom:5px;">
            Predicted YPH: {predicted_yph:.2f}
        </div>
        """, unsafe_allow_html=True)

        # --- Display AI Explanation ---
        st.markdown(f"""
        <div style="color:#1B4332; font-weight:bold; font-size:1.3rem; margin-top:10px; margin-bottom:5px;">
            Explanation
        </div>
        <div style="background-color:white; color:black; border:1px solid #d3d3d3; border-radius:6px; padding:14px; font-size:1rem; line-height:1.4;">
        {explanation_text}
        </div><br>
        """, unsafe_allow_html=True)

        # --- Download PDF Report ---
        pdf_buffer, filename = generate_pdf_report(
            T200, U709, Epson, Dolomite, Urea, BioShield,
            temperature_2m_mean, rain_sum, wind_speed_10m_mean, et0_fao_evapotranspiration,
            Dolomite_moisture, YPH_lag1, YPH_lag12, Compost_lag1,
            predicted_yph, explanation_text, year=year, month=month
        )

        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name=filename,
            mime="application/pdf"
        )

    st.markdown('</div>', unsafe_allow_html=True)

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

