import streamlit as st
from openai import OpenAI
import time
import PyPDF2
import io

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
    else:
        st.warning("Could not extract content from the file.")


# === Page Configuration ===
st.set_page_config(
    page_title="Groq LLaMA 3 Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Sidebar: Only Clear Chat Button ===
with st.sidebar:
    st.header("Chat Controls")
    if st.button("Clear Chat"):
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
        ]
        st.session_state.input_box = ""  # Also clear input



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
        st.markdown(f"**Assistant:** {message['content']}")

# === User Input ===
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message here...")
    submitted = st.form_submit_button("Send")

if submitted and groq_api_key and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.markdown(f"**You:** {user_input}")

    try:
        # Initialize OpenAI client with Groq API
        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        # Call the model with streaming
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=st.session_state.chat_history,
            stream=True
        )

        # Typing animation
        assistant_response = ""
        response_container = st.empty()
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            assistant_response += content
            # Simulate typing effect
            response_container.markdown(f"**Assistant:** {assistant_response}")
            time.sleep(0.02)

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    except Exception as e:
        st.error(f"Error: {str(e)}")

elif not groq_api_key:
    st.warning("Please provide your Groq API Key in Streamlit secrets to start chatting.")
