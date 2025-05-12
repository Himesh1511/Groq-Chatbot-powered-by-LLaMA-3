import streamlit as st
from openai import OpenAI
import time
import PyPDF2
import io

# === Page Config ===
st.set_page_config("Groq LLaMA 3 Chatbot", layout="wide", initial_sidebar_state="expanded")

# === Initialize Chat History ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
    ]

# === Sidebar ===
with st.sidebar:
    st.header("Controls")

    if st.button("Clear Chat"):
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
        ]
        st.session_state.pop("repeat_message", None)
        st.session_state.pop("edited_message", None)

    if st.button("Repeat Last Message"):
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "user":
                st.session_state.repeat_message = msg["content"]
                break

    if st.button("Edit Last Message"):
        for i in reversed(range(len(st.session_state.chat_history))):
            if st.session_state.chat_history[i]["role"] == "user":
                st.session_state.edited_message = st.session_state.chat_history[i]["content"]
                st.session_state.chat_history.pop(i)
                break

    st.subheader("Upload File")
    uploaded_file = st.file_uploader("Upload a PDF or TXT", type=["pdf", "txt"])
    if uploaded_file:
        file_text = ""
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                file_text += page.extract_text() or ""
        else:
            file_text = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()

        if file_text.strip():
            st.success("File content loaded.")
            st.session_state.chat_history.append({
                "role": "system",
                "content": f"The user uploaded the following document:\n{file_text[:4000]}"
            })
        else:
            st.warning("Couldn't extract content.")

# === Title ===
st.title("Groq Chatbot powered by LLaMA 3")

# === Load API Key ===
groq_api_key = st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.warning("Set your Groq API Key in secrets to chat.")
    st.stop()

# === Message Renderer ===
def render_chat():
    for msg in st.session_state.chat_history:
        if msg["role"] == "system":
            continue
        with st.container():
            if msg["role"] == "user":
                st.markdown(
                    f"""<div style="text-align: right; background-color: #dcf8c6; padding: 0.7em; border-radius: 10px; margin: 5px 0;">{msg["content"]}</div>""",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div style="text-align: left; background-color: #f1f0f0; padding: 0.7em; border-radius: 10px; margin: 5px 0;">{msg["content"]}</div>""",
                    unsafe_allow_html=True
                )

render_chat()

# === Handle Message Input ===
input_prompt = "Type your message here..."
prefilled = ""

if "edited_message" in st.session_state:
    input_prompt = "Editing your last message..."
    prefilled = st.session_state.pop("edited_message")
elif "repeat_message" in st.session_state:
    prefilled = st.session_state.pop("repeat_message")

user_input = st.chat_input(input_prompt, key="chat_input")
if prefilled and not user_input:
    user_input = prefilled

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    render_chat()  # Immediately show user message

    with st.spinner("Assistant is typing..."):
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
            assistant_reply = ""
            response_placeholder = st.empty()
            for chunk in response:
                token = chunk.choices[0].delta.content or ""
                assistant_reply += token
                response_placeholder.markdown(
                    f"""<div style="text-align: left; background-color: #f1f0f0; padding: 0.7em; border-radius: 10px; margin: 5px 0;">{assistant_reply}</div>""",
                    unsafe_allow_html=True
                )
                time.sleep(0.01)

            st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

        except Exception as e:
            st.error(f"Error: {e}")
