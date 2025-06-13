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
            # Reduce PDF size for context window
            st.session_state['uploaded_file_text'] = file_text[:1000]
        else:
            st.warning("Could not extract content from the file.")

# === Title ===
st.title("Groq Chatbot powered by LLaMA 3")

# === Show Streamlit version for debugging ===
st.write("Streamlit version:", st.__version__)

# === Load Groq API Key ===
groq_api_key = st.secrets.get("GROQ_API_KEY")

# === Initialize Chat History ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
    ]

# === Display Chat History ===
st.write("### Chat")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(
            f"""
<div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
    <div style='max-width: 70%; background-color: #d4edda; padding: 10px 14px; border-radius: 12px;'>
        <div style='font-weight: bold; margin-bottom: 4px; color: #155724;'>You</div>
        <div style='word-wrap: break-word;'>{message['content']}</div>
    </div>
</div>
            """,
            unsafe_allow_html=True
        )
    elif message["role"] == "assistant":
        st.markdown(
            f"""
<div style='display: flex; justify-content: flex-start; margin: 10px 0;'>
    <div style='max-width: 70%; background-color: #f1f0f0; padding: 10px 14px; border-radius: 12px;'>
        <div style='font-weight: bold; margin-bottom: 4px; color: #333;'>Assistant</div>
        <div style='word-wrap: break-word;'>{message['content']}</div>
    </div>
</div>
            """,
            unsafe_allow_html=True
        )

# === Chat Input ===
input_prompt = "Type your message here..."
default_input = None

if "edited_message" in st.session_state:
    input_prompt = "Editing your last message..."
    default_input = st.session_state.pop("edited_message")

if "repeat_message" in st.session_state:
    default_input = st.session_state.pop("repeat_message")

user_input = st.chat_input(input_prompt, key="chat_input")
if user_input is None and default_input:
    st.session_state.pending_input = default_input
    st.rerun()

if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

# === Handle Input (User) ===
if user_input and groq_api_key:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.rerun()

# DEBUG output
st.write("DEBUG: chat_history", st.session_state.chat_history)
if 'uploaded_file_text' in st.session_state:
    st.write("DEBUG: uploaded_file_text exists and length", len(st.session_state['uploaded_file_text']))

# === Generate Assistant Response ===
if (
    groq_api_key 
    and len(st.session_state.chat_history) > 0 
    and st.session_state.chat_history[-1]["role"] == "user" 
    and (len(st.session_state.chat_history) < 2 or st.session_state.chat_history[-2]["role"] != "assistant")
):
    messages = st.session_state.chat_history.copy()
    if 'uploaded_file_text' in st.session_state:
        messages.insert(1, {
            "role": "system",
            "content": f"A document has been uploaded by the user. Here is the content (truncated):\n{st.session_state['uploaded_file_text'][:1000]}"
        })
    st.write("DEBUG: messages for LLM", messages)
    try:
        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            stream=True
        )
        assistant_response = ""
        response_container = st.empty()
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            assistant_response += content
            response_container.markdown(
                f"<div style='text-align: left; background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin: 5px 0;'><strong>Assistant:</strong> {assistant_response}</div>",
                unsafe_allow_html=True)
            time.sleep(0.02)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
        st.rerun()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("DEBUG: Exception occurred:", e)

elif not groq_api_key:
    st.warning("Please provide your Groq API Key in Streamlit secrets to start chatting.")
