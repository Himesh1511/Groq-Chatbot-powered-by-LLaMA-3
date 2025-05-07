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

# === Sidebar: Chat Controls ===
with st.sidebar:
    st.header("Chat Controls")
    if st.button("Clear Chat"):
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
        ]
        st.session_state.input_box = ""
    if st.button("Repeat Last Message"):
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            st.session_state.input_box = st.session_state.chat_history[-1]["content"]
    if st.button("Edit Last Message"):
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            st.session_state.input_box = st.session_state.chat_history[-1]["content"]

    # === File Upload Section ===
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
            # Add summarization to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": "Summarize the following document:\n" + file_text[:3000]
            })
        else:
            st.warning("Could not extract content from the file.")

# === Title ===
st.title("Groq Chatbot powered by LLaMA 3")

# === Load Groq API Key ===
groq_api_key = st.secrets["GROQ_API_KEY"]

# === Initialize Chat History ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
    ]

# === Display Chat History ===
st.write("### Chat")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        st.markdown(f"**Assistant:**\n\n{message['content']}", unsafe_allow_html=True)

# === User Input (Locked at bottom) ===
st.markdown("""
<style>
.stForm {position: fixed; bottom: 0; background: white; width: 100%; padding: 1rem; z-index: 999;}
</style>
""", unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    default_val = st.session_state.get("input_box", "")
    user_input = st.text_input("Type your message here...", value=default_val, key="input_box_form")
    submitted = st.form_submit_button("Send")

if submitted and groq_api_key and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.markdown(f"**You:** {user_input}")

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
