import streamlit as st
from pathlib import Path
import base64

# Page Config
BASE_DIR = Path(__file__).resolve().parent.parent
favicon_path = BASE_DIR / "assets" / "favicon_zoom.ico"

st.set_page_config(
    page_title="GPTea Home",
    page_icon=str(favicon_path),
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
logo_path = BASE_DIR / "assets" / "logo_white_cropped.png"

# Convert logo to base64
with open(logo_path, "rb") as img_file:
    logo_base64 = base64.b64encode(img_file.read()).decode()

# Custom CSS
st.markdown("""
<style>

/* Hide Streamlit default header safely */
header[data-testid="stHeader"] {
    display: none;
}
.stVerticalBlock {
        gap: 0rem !important;
    }
/* Remove extra padding without breaking layout */
.block-container {
    padding-top: 0rem;
    padding-left: 0rem;
    padding-right: 0rem;
    padding-bottom: 0rem;
    max-width: 100%;
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}

/* App background + default text color */
.stApp {
    background-color: white;
    color: black;
}

/* Force all normal text to black */
p, span, div, label {
    color: black;
}

/* Headings */
h1, h2, h3 {
    color: #1B4332;
    font-weight: bold;
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

.full-width-img {
    width: 100%;
}

.circle-img {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    object-fit: cover;
    margin-bottom: 15px;
    align: center;
}

/* Navbar */
.header {
    position: sticky;
    top: 0;
    z-index: 999;
    background-color: #1B4332;
    padding: 10px 10px;       /* remove top/bottom padding, keep left/right */
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
    height: 70px;          /* fixed navbar height */
}

/* Navbar links */
.header a {
    color: white;
    text-decoration: none;
    margin-left: 20px;
    font-weight: bold;
}

.header a:hover {
    color: #D8F3DC;
}

/* Cards */
.card {
    background-color: #D8F3DC;
    display: flex !important;
    flex-direction: column !important;  
    justify-content: center !important;
    align-items: center !important;     
    padding: 20px;
    border-radius: 15px;
    text-align: left;
    transition: transform 0.3s ease;
}

.card:hover {
    transform: scale(1.05);
}

.section-card {
    background-color: #D8F3DC; 
    padding: 20px;
    border-radius: 15px;
    text-align: left;
    transition: transform 0.3s ease;
    font-size:18px !important;
    line-height: 1.6;
}

.section-card:hover {
    transform: scale(1.05);
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

.logo {
    display: block;
    height: 100%;           
    width: auto;
    object-fit: contain;
    margin: 0;
    padding: 0;
}

.profile-card {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    background-color: #D8F3DC;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    transition: transform 0.3s ease;
    min-height: 280px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.profile-card:hover {
    transform: scale(1.05);
}

.profile-card img {
    margin-bottom: 15px;
}

.profile-card h4 {
    color: #1B4332;
    margin: 10px 0 5px 0;
}

.profile-card p {
    color: #666;
    margin: 0 0 10px 0;
}

.icon-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 10px;
}

/* Style the Streamlit button to match */
div.stButton > button {
    background: linear-gradient(45deg, #1B4332, #2D5A44);
    color: white !important;
    border: none !important;
    padding: 14px 24px !important;
    font-weight: bold !important;
    font-size: 25px !important;
    height: 50px !important;
    margin-top: 10px;
}

/* Target the ACTUAL button text element */
div[data-testid="stButton"] > button label,
div[data-testid="stButton"] button div[data-testid="stMarkdownContainer"] p,
div.stButton > button > span,
div.stButton > button * {
    color: white !important;
    font-weight: 700 !important;
    font-size: 18px !important;
    letter-spacing: 0.5px !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
}

.stButton > button:hover {
    background: linear-gradient(45deg, #2D5A44, #1B4332) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(27,67,50,0.4) !important;
}

.font-size {
    font-size:18px !important;
    line-height: 1.6;
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


# Hero Section
hero_image_path = BASE_DIR / "assets" / "homepage_intro_image.png"
with open(hero_image_path, "rb") as f:
    hero_bytes = f.read()
    hero_base64 = base64.b64encode(hero_bytes).decode()

# Display hero image with full width and controlled height
st.markdown(f"""
<img src="data:image/png;base64,{hero_base64}" 
     style="width:100vw; height:675px; object-fit:cover; display:block; margin:0; padding:0;">
""", unsafe_allow_html=True)

col1, col_left_gap, col_main, col_right_gap, col5 = st.columns([0.25, 0.25, 7, 0.25, 0.25])
with col_main:
    # What is GPTea Section
    st.markdown("<a name='what'></a>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h2>What is GPTea?</h2><br>", unsafe_allow_html=True)


    st.write('<div class="font-size"> GPTea is an AI-powered smart agriculture system created to support tea farmers. Tea farming faces many challenges such as unpredictable weather, plant diseases, and unstable production levels. GPTea uses data and machine learning to help solve these problems. <br><br>The system has three main components. First, it predicts future tea yield using historical production and environmental data. This helps farmers plan better and make informed decisions. Second, it detects tea leaf diseases using image analysis, allowing farmers to identify problems early and reduce crop loss. Third, GPTea includes a chatbot that answers tea-related questions, providing quick guidance and support anytime. By combining prediction, detection, and intelligent assistance, GPTea aims to improve productivity, reduce losses, and promote smarter and more sustainable tea farming.</div>', unsafe_allow_html=True)


    # Components Section
    st.markdown("<a name='components'></a>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h2>Components</h2><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # Component Image 1
    component_1_path = BASE_DIR / "assets" / "component_1.png"
    with open(component_1_path, "rb") as f:
        component_1_b64 = base64.b64encode(f.read()).decode()

    # Component Image 1
    component_2_path = BASE_DIR / "assets" / "component_2.png"
    with open(component_2_path, "rb") as f:
        component_2_b64 = base64.b64encode(f.read()).decode()

    # Component Image 1
    component_3_path = BASE_DIR / "assets" / "component_3.png"
    with open(component_3_path, "rb") as f:
        component_3_b64 = base64.b64encode(f.read()).decode()

    with col1:
        with st.container():
            st.markdown(f"""
            <div class="card">
                <img src="data:image/png;base64,{component_1_b64}" alt="Check Yield" class="full-width-img">
                <p style="font-weight: bold; font-size: 22px; margin-top: 15px; margin-bottom: 5px;color:#1B4332;"> 
                    Tea Yield Prediction System
                </p>
                <p style="color: #666;">
                Input Weather and Fertilizer Data and get the predicted Yield Per Hectare with explanations.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Button now appears INSIDE the same section visually
            if st.button("Check Yield Now", key="yield_btn", use_container_width=True):
                st.switch_page("C:\\Users\\kavis\\PycharmProjects\\GPTea\\pages\\02_tea_yield.py")

    with col2:
        with st.container():
            st.markdown(f"""
            <div class="card">
                <img src="data:image/png;base64,{component_2_b64}" alt="Detect Disease" class="full-width-img" style="width: 100%; height: 170px; object-fit: cover;">
                <p style="font-weight: bold; font-size: 22px; margin-top: 15px; margin-bottom: 5px;color:#1B4332;">
                    Tea Leaf Disease Detection System</p>
                <p style="color: #666;">
                Input the image of the diseased leaf and get the disease detected with explanations.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Button now appears INSIDE the same section visually
            if st.button("Detect Disease Now", key="disease_btn", use_container_width=True):
                st.switch_page("C:\\Users\\kavis\\PycharmProjects\\GPTea\\pages\\03_disease_detection.py")

    with col3:
        with st.container():
            st.markdown(f"""
            <div class="card">
                <img src="data:image/png;base64,{component_3_b64}" alt="Chatbot" class="full-width-img" style="width: 100%; height: 170px; object-fit: cover;">
                <p style="font-weight: bold; font-size: 22px; margin-top: 15px; margin-bottom: 5px;color:#1B4332;">
                        Chatbot System</p>
                <p style="color: #666;">
                Chat with our specialized AI agent to clarify any doubts regarding tea industry.
                </p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Chat Now", key="chat_btn", use_container_width=True):
                st.switch_page("")

    # Research Innovation Section
    st.markdown("<a name='innovation'></a>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h2>Research Innovation</h2><br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1.5])
    with col1:
        st.write('<div class="font-size">GPTea uses a Ridge Regression model selected after comparing six machine learning models for yield prediction. Walk-forward validation and lag features were applied to ensure accurate and reliable time-series yield prediction.<br>This disease detection model leverages a CNN with MobileNet architecture to enable efficient, high-accuracy image classification, supporting fast, scalable, and reliable medical diagnosis.<br>This model uses a RAG framework with an LLM to generate context-grounded insights, reducing hallucinations through real-time data retrieval and ensuring accurate, auditable market analysis.</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="section-card">
        <h3>Key Achievements</h3>
        1. Developed a Ridge Regression yield prediction model after comparing six machine learning models to select the best-performing approach.<br>
        2. Applied walk-forward validation and lag features to ensure accurate and reliable time-series prediction, achieving 92% test accuracy.<br>
        3. Lightweight, accurate disease classification using MobileNet.<br>
        4. Built a RAG-powered LLM delivering accurate, real-time consumer insights with reduced hallucinations.
        </div>
        """, unsafe_allow_html=True)

    # Research Team Section
    st.markdown("<a name='team'></a>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h2>Research Team</h2><br>", unsafe_allow_html=True)

    col1, gap_col_1, col2, gap_col_2, col3  = st.columns([1,0.1,1,0.1,1])

    # Load all team member images first
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Kavishka
    kavishka_path = BASE_DIR / "assets" / "kavishka_profile.png"
    with open(kavishka_path, "rb") as f:
        kavishka_b64 = base64.b64encode(f.read()).decode()

    # Sharubini
    sharubini_path = BASE_DIR / "assets" / "sharubini_profile.jpeg"
    with open(sharubini_path, "rb") as f:
        sharubini_b64 = base64.b64encode(f.read()).decode()

    # Kavindu
    kavindu_path = BASE_DIR / "assets" / "kavindu_profile.jpeg"
    with open(kavindu_path, "rb") as f:
        kavindu_b64 = base64.b64encode(f.read()).decode()

    team_members = [
        ("Kavishka Karunagaran", "Yield Prediction System", kavishka_b64,
         "https://github.com/Kavishka2401",
         "https://www.linkedin.com/in/kavishka-karunagaran-a76a82332/"),

        ("Sharubini Sathasivam", "Pest Detection System", sharubini_b64,
         "https://github.com/Sharubini2",
         "https://www.linkedin.com/in/sharubini-sathasivam-b8a621333/"),

        ("Kavindu Sudasinghe", "Chatbot System", kavindu_b64,
         "https://github.com/Kavindu-20241240",
         "https://www.linkedin.com/in/kavindu-sudasinghe-591b47397/")
    ]

    cols = st.columns(3)

    for idx, (name, role, img_b64, github, linkedin) in enumerate(team_members):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="profile-card">
                    <img src="data:image/png;base64,{img_b64}" alt="{name}" style="width:120px; height:120px; border-radius:50%; object-fit:cover;">
                    <h4>{name}</h4>
                    <p>{role}</p>
                    <div class="icon-row">
                        <a href="{github}" target="_blank">
                            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="25">
                        </a>
                        <a href="{linkedin}" target="_blank">
                            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25">
                        </a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

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

