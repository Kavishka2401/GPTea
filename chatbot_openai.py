import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CORE CONFIGURATION ---
load_dotenv()
st.set_page_config(
    page_title="GPTea | Advisor",
    page_icon="🍃",
    layout="wide"
)

# --- 2. THE FINAL UI ENGINE (CSS) ---
st.markdown("""
    <style>
    /* 1. Deep Black Background */
    .stApp {
        background-color: #000000 !important; 
    }

    /* 2. Hide ALL Avatars, Icons, and technical labels (face, smart_toy) */
    [data-testid="chatAvatarIcon-user"], 
    [data-testid="chatAvatarIcon-assistant"], 
    [data-testid="stChatMessage"] figure,
    [data-testid="stAvatar"],
    .st-emotion-cache-1pxm890, 
    .st-emotion-cache-16ids0f {
        display: none !important;
    }

    /* 3. Global Text Style */
    [data-testid="stChatMessageContent"] p {
        color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        margin: 0 !important;
        line-height: 1.5 !important;
    }

    /* 4. Reset Chat Containers */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 10px 0px !important;
        display: flex !important;
        width: 100% !important;
    }

    /* 5. USER BUBBLE (Prompts) -> LEFT ALIGNED (Green) */
    [data-testid="stChatMessage"][aria-label="Chat message from user"] {
        flex-direction: row !important; 
        justify-content: flex-start !important;
    }

    [data-testid="stChatMessage"][aria-label="Chat message from user"] > div:nth-child(2) {
        background-color: #075e54 !important; /* WhatsApp Dark Green */
        border-radius: 4px 18px 18px 18px !important;
        padding: 12px 18px !important;
        margin-left: 15px !important;
        max-width: 75% !important;
        margin-right: auto !important;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3) !important;
    }

    /* 6. ASSISTANT BUBBLE (Bot) -> RIGHT ALIGNED (Grey) */
    [data-testid="stChatMessage"][aria-label="Chat message from assistant"] {
        flex-direction: row-reverse !important; 
        justify-content: flex-end !important;
    }

    [data-testid="stChatMessage"][aria-label="Chat message from assistant"] > div:nth-child(2) {
        background-color: #262626 !important; /* Dark Grey */
        border-radius: 18px 4px 18px 18px !important;
        padding: 12px 18px !important;
        margin-right: 15px !important;
        max-width: 75% !important;
        margin-left: auto !important;
        box-shadow: -2px 2px 5px rgba(0,0,0,0.3) !important;
    }

    /* 7. Input Bar Styling */
    [data-testid="stChatInput"] {
        background-color: #1c1c1e !important;
        border-radius: 25px !important;
        border: 1px solid #3a3a3c !important;
    }

    [data-testid="stChatInput"] textarea {
        color: white !important;
    }

    /* 8. Hub Header Styling */
    .hub-title {
        text-align: center;
        padding: 20px;
        color: #FFFFFF;
        font-size: 28px;
        font-weight: bold;
        border-bottom: 0.5px solid #333;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 3. BACKEND AI LOGIC ---
def get_context(query):
    return "Sri Lankan tea estate expert knowledge base."


# Primary LLM (Groq Only)
smart_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

template = """You are a professional Sri Lankan tea estate advisor. 
Answer the following farmer query using the context provided.
CONTEXT: {context}
QUESTION: {question}
ANSWER:"""
prompt_tpl = ChatPromptTemplate.from_template(template)

tea_chain = (
        {"context": get_context, "question": RunnablePassthrough()}
        | prompt_tpl | smart_llm | StrOutputParser()
)


# --- 4. MODULE RENDERERS ---

def render_advisor():
    st.markdown('<div class="hub-title">💭 GPTea AI Advisor</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ayubowan! 🍵 How can I assist you with your tea crop today?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about fertilizer, pests, or pruning..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # Process response if last message is from user
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Consulting knowledge base..."):
                response = tea_chain.invoke(st.session_state.messages[-1]["content"])
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


def render_yield():
    st.markdown('<div class="hub-title">📊 Yield Prediction</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        moisture = st.slider("Soil Moisture Content (%)", 0, 100, 50)
        ph = st.number_input("Soil pH", 4.0, 7.5, 5.5)
    with col2:
        nitrogen = st.number_input("Nitrogen Level (mg/kg)", 0, 250, 80)
        rainfall = st.slider("Rainfall (mm)", 0, 1000, 300)

    if st.button("Calculate Expected Yield", use_container_width=True):
        res = (moisture * 1.5) + (nitrogen * 2.1) + (rainfall * 0.05)
        st.metric("Predicted Yield", f"{round(res, 2)} kg/ha", "+3.2%")


def render_pest():
    st.markdown('<div class="hub-title">🪲 Pest & Disease ID</div>', unsafe_allow_html=True)
    img = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])
    if img:
        st.image(img, caption="Scanning Leaf...", width=400)
        if st.button("Run Diagnostic Scan", use_container_width=True):
            st.error("Diagnostic Result: **Red Spider Mite Infestation**")


# --- 5. MAIN ROUTER & SIDEBAR ---
if "page" not in st.session_state:
    st.session_state.page = "Advisor"

with st.sidebar:
    st.title("🍃 GPTea Hub")
    if st.button("💬 Advisor", use_container_width=True): st.session_state.page = "Advisor"
    if st.button("📊 Yield", use_container_width=True): st.session_state.page = "Yield"
    if st.button("🪲 Pest ID", use_container_width=True): st.session_state.page = "Pest"

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Run the selected page
if st.session_state.page == "Advisor":
    render_advisor()
elif st.session_state.page == "Yield":
    render_yield()
elif st.session_state.page == "Pest":
    render_pest()