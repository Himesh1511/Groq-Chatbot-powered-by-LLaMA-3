import streamlit as st
from openai import OpenAI
import time
import PyPDF2
import io

st.set_page_config(
    page_title="Groq LLaMA 3 Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.header("Chat Controls")

    if st.button("Clear Chat"):
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
        ]
        st.session_state.pop("repeat_message", None)
        st.session_state.pop("edited_message", None)
        st.session_state.pop("uploaded_file_text", None)

    if st.button("Repeat Last Message"):
        if st.session_state.get("chat_history"):
            last_user = next(
                (msg["content"] for msg in reversed(st.session_state.chat_history) if msg["role"] == "user"), None)
            if last_user:
                st.session_state.repeat_message = last_user

    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Upload a .pdf or .txt file", type=["pdf", "txt"])
    file_text = ""

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                file_text += page.extract_text() or ""
        elif uploaded_file.type == "text/plain":
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            file_text = stringio.read()

        if file_text.strip():
            st.success("File uploaded and content loaded.")
            st.session_state['uploaded_file_text'] = file_text[:1000]  # reduced for context window
        else:
            st.warning("Could not extract content from the file.")

st.title("Groq Chatbot powered by LLaMA 3")
st.write("Streamlit version:", st.__version__)

groq_api_key = st.secrets.get("GROQ_API_KEY")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
    ]

st.write("### Chat")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                <
