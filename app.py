import streamlit as st
from openai import OpenAI
import time
import PyPDF2
import io

# === Page Configuration ===
st.set_page_config(
    page_title="Groq LLaMA 3 Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Load Groq API Key ===
groq_api_key = st.secrets["GROQ_API_KEY"]

# === Initialize Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
    ]
if "input_box" not in st.session_state:
    st.session_state.input_box = ""

# === Sidebar: Controls and File Upload ===
with st.sidebar:
    st.header("Chat Controls")
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
        ]
        st.session_state.input_box = ""

    if st.button("Repeat Last Message"):
        # Re-send the last user message
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "user":
                st.session_state.input_box = msg["content"]
                break

    if st.button("Edit Last Message"):
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "user":
                st.session_state.input_box = msg["content"]
                break

    st.markdown("---")
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
            st.session_state.chat_history.append({
                "role": "user",
                "content": "Summarize the following document:\n" + file_text[:3000]
            })
        else:
            st.warning("Could not extract content from the file.")

# === Title ===
st.title("Groq Chatbot powered by LLaMA 3")

# === Display Chat History ===
st.write("### Chat")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        st.markdown(f"**Assistant:**\n\n{message['content']}", unsafe_allow_html=True)

# === Fixed Bottom Input Box ===
st.markdown("""
    <style>
    .bottom-form-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: white;
        padding: 1rem 2rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
        z-index: 9999;
    }
    .block-container {
        padding-bottom: 200px !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="bottom-form-container">', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message here...", value=st.session_state.input_box, key="input_box")
        submitted = st.form_submit_button("Send")
    st.markdown('</div>', unsafe_allow_html=True)

# === Handle User Submission ===
if submitted and groq_api_key and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.input_box = ""  # Clear input field

    try:
        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=st.session_state.chat_history,
            stream=True
        )

        assistant_response = ""
        response_container = st.empty()
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            assistant_response += content
            response_container.markdown(f"**Assistant:**\n\n{assistant_response}", unsafe_allow_html=True)
            time.sleep(0.02)

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    except Exception as e:
        st.error(f"Error: {str(e)}")

elif not groq_api_key:
    st.warning("Please provide your Groq API Key in Streamlit secrets to start chatting.")
