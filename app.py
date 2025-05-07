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

# === Sidebar: Controls ===
with st.sidebar:
    st.header("Chat Controls")

    if st.button("Clear Chat"):
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
        ]

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

# === Title ===
st.title("Groq Chatbot powered by LLaMA 3")

# === Load Groq API Key ===
groq_api_key = st.secrets.get("GROQ_API_KEY")

# === Initialize Chat History ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
    ]

# === File Upload Section ===
st.subheader("Document Q&A")
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
        # Add the file text as system context
        st.session_state.chat_history.append({
            "role": "system",
            "content": f"The following document has been uploaded by the user. Use it to answer questions:\n{file_text[:4000]}"
        })
    else:
        st.warning("Could not extract content from the file.")

# === Display Chat History ===
st.write("### Chat")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        st.markdown(f"**Assistant:** {message['content']}")

# === Chat Input ===
user_input = st.chat_input("Type your message here...")

# === Use repeated or edited message if available ===
if "repeat_message" in st.session_state:
    user_input = st.session_state.pop("repeat_message")

if "edited_message" in st.session_state:
    user_input = st.chat_input("Edit your last message...", value=st.session_state.pop("edited_message"))

# === Generate Assistant Response ===
if user_input and groq_api_key:
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
            response_container.markdown(f"**Assistant:** {assistant_response}")
            time.sleep(0.02)

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    except Exception as e:
        st.error(f"Error: {str(e)}")

elif not groq_api_key:
    st.warning("Please provide your Groq API Key in Streamlit secrets to start chatting.")
