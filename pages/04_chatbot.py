import base64
from pathlib import Path
import streamlit as st
from utils.rag_retrieval import retrieve_docs
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pages.feedback_utils import render_feedback_widget

# Show initial message from other pages
if "chat_init_message" in st.session_state:
    init_msg = st.session_state.pop("chat_init_message")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": init_msg})

# Page Config
BASE_DIR = Path(__file__).resolve().parent.parent
favicon_path = BASE_DIR / "assets" / "favicon_zoom.ico"
logo_path    = BASE_DIR / "assets" / "logo_white_cropped.png"

with open(logo_path, "rb") as img_file:
    logo_base64 = base64.b64encode(img_file.read()).decode()

st.set_page_config(
    page_title="Tea Advisor",
    page_icon=str(favicon_path),
    layout="wide"
)

# UI Styling
st.markdown("""
<style>
/* Force Entire Page White */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #FFFFFF !important;
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

.block-container {
    padding: 0rem !important;
    max-width: 100% !important;
}

/* Chat Container */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 10px 0px !important;
    display: flex !important;
    width: 100% !important;
}

/* User Bubble */
[data-testid="stChatMessage"][aria-label="Chat message from user"] > div:nth-child(2) {
    background-color: #DCF8E4 !important;
    border-radius: 4px 18px 18px 18px !important;
    padding: 12px 18px !important;
    margin-left: 15px !important;
    max-width: 75% !important;
    margin-right: auto !important;
}

/* Assistant Bubble */
[data-testid="stChatMessage"][aria-label="Chat message from assistant"] > div:nth-child(2) {
    background-color: #F1F1F1 !important;
    border-radius: 18px 4px 18px 18px !important;
    padding: 12px 18px !important;
    margin-right: 15px !important;
    max-width: 75% !important;
    margin-left: auto !important;
}

/* Force all chat text to black */
[data-testid="stChatMessage"] * {
    color: #000000 !important;
}

/* Chat input area */
[data-testid="stChatInput"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #D8F3DC !important;
    border: 2px solid #1B4332 !important;
    border-radius: 12px !important;
    color: #1B4332 !important;
    padding: 10px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #1B433280 !important;
}

[data-testid="stChatInput"] button {
    background-color: #D8F3DC !important;
    border: 1px solid #1B4332 !important;
    border-radius: 8px !important;
    right: 10px !important;
}

[data-testid="stChatInput"] button svg {
    fill: #1B4332 !important;
}

[data-testid="stChatInput"] textarea:focus {
    box-shadow: 0 0 5px #1B4332 !important;
    outline: none !important;
}

footer {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"""
<div class="header">
    <img src="data:image/png;base64,{logo_base64}" class="logo">
    <div style="display:flex; gap:10px;">
        <a href="/home" target="_self">Home</a>
        <a href="/tea_yield" target="_self">Yield Prediction Page</a>
        <a href="/disease_detection" target="_self">Disease Detection Page</a>
        <a href="/chatbot" target="_self">Chatbot Page</a>
    </div>
</div>
""", unsafe_allow_html=True)

#FEEDBACK
render_feedback_widget(
    page_id   = "chatbot",
    icon_path = str(BASE_DIR / "assets" / "favicon_zoom.png"),
)

col1, col_left_gap, col_main, col_right_gap, col5 = st.columns([0.25, 0.25, 7, 0.25, 0.25])

# Context Retrieval
def get_context(query: str) -> str:
    """Fetch knowledge base context via RAG."""
    docs = retrieve_docs(query, k=3)
    if isinstance(docs, dict):
        return docs.get("message", "No relevant context found in knowledge base.")
    context_parts = []
    for doc in docs:
        context_parts.append(f"Q: {doc['question']}\nA: {doc['answer']}")
    return "\n\n---\n\n".join(context_parts)

# LLM Chain
smart_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key="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"
)

template = """You are a Sri Lankan tea estate advisor AI.
Answer the query using ONLY the context provided below.
If the context is insufficient, say so clearly.
CONTEXT: {context}
QUESTION: {question}
ANSWER:"""
prompt_tpl = ChatPromptTemplate.from_template(template)

tea_chain = (
    {"context": get_context, "question": RunnablePassthrough()}
    | prompt_tpl
    | smart_llm
    | StrOutputParser()
)

# Chat Rendering
with col_main:
    st.title("Tea Advisor")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I assist you with your tea crop today?"}
        ]

    # Render all existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Generate assistant reply for the latest user message
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Consulting knowledge base..."):
                response = tea_chain.invoke(st.session_state.messages[-1]["content"])
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

# New Input
if prompt := st.chat_input("Ask about fertilizer, pests, pruning, soil..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()