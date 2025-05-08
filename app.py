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

# === Sidebar: Controls and File Upload ===
with st.sidebar:
    st.header("Chat Controls")

    if st.button("Clear Chat"):
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
        ]
        st.session_state.pop("repeat_message", None)
        st.session_state.pop("edited_message", None)

    if st.button("Repeat Last Message"):
        if st.session_state.get("chat_history"):
            last_user = next(
                (msg["content"] for msg in reversed(st.session_state.chat_history) if msg["role"] == "user"), None)
            if last_user:
                st.session_state.repeat_message = last_user

    if st.button("Edit Last Message"):
        if st.session_state.get("chat_history"):
            for i in reversed(range(len(st.session_state.chat_history))):
                if st.session_state.chat_history[i]["role"] == "user":
                    st.session_state.edited_message = st.session_state.chat_history[i]["content"]
                    st.session_state.chat_history.pop(i)
                    break

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
            st.session_state.chat_history.append({
                "role": "system",
                "content": f"The following document has been uploaded by the user. Use it to answer questions:\n{file_text[:4000]}"
            })
        else:
            st.warning("Could not extract content from the file.")

# === Title ===
st.title("Groq Chatbot powered by LLaMA 3")

# === Load Groq API Key ===
groq_api_key = st.secrets.get("GROQ_API_KEY")

# === Initialize Chat History ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
    ]

# === Display Chat History with Chat Bubbles ===
chat_css = """
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 1rem 0;
}
.chat-message {
    max-width: 80%;
    padding: 0.75rem 1rem;
    border-radius: 1rem;
    line-height: 1.5;
    word-wrap: break-word;
}
.user-message {
    align-self: flex-end;
    background-color: #DCF8C6;
    text-align: right;
}
.assistant-message {
    align-self: flex-start;
    background-color: #F1F0F0;
    text-align: left;
}
</style>
<div class="chat-container">
"""
for msg in st.session_state.chat_history:
    if msg["role"] == "system":
        continue
    role_class = "user-message" if msg["role"] == "user" else "assistant-message"
    chat_css += f'<div class="chat-message {role_class}">{msg["content"]}</div>'
chat_css += "</div>"
st.markdown(chat_css, unsafe_allow_html=True)

# === Chat Input ===
input_prompt = "Type your message here..."
default_input = None

if "edited_message" in st.session_state:
    input_prompt = "Editing your last message..."
    default_input = st.session_state.pop("edited_message")

elif "repeat_message" in st.session_state:
    default_input = st.session_state.pop("repeat_message")

user_input = st.chat_input(input_prompt)

if user_input is None and default_input:
    st.warning(f"{input_prompt} (copied below for editing):")
    user_input = st.text_area("Edit below and press Enter:", default_input, height=100)

# === Generate Assistant Response ===
if user_input and groq_api_key:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

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
            chat_bubble = f"""
            <div class="chat-container">
                <div class="chat-message assistant-message">{assistant_response}</div>
            </div>
            """
            response_container.markdown(chat_bubble + "<style>" + chat_css + "</style>", unsafe_allow_html=True)
            time.sleep(0.02)

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    except Exception as e:
        st.error(f"Error: {str(e)}")

elif not groq_api_key:
    st.warning("Please provide your Groq API Key in Streamlit secrets to start chatting.")
