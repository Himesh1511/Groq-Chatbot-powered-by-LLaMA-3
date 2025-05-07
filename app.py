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

# === Sidebar ===
with st.sidebar:
    st.header("Chat Controls")

    # File upload
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
    # === Repeat Button ===
if st.button("🔁 Repeat Last Message"):
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        st.session_state.input_box = st.session_state.chat_history[-1]["content"]
    else:
        st.warning("No previous user message to repeat.")
            

    # Clear chat button
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
        ]
        st.session_state.input_box = ""


# === Page Title ===
st.title("Groq Chatbot powered by LLaMA 3")

# === API Key ===
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

# === Styling for fixed footer input ===
st.markdown("""
<style>
/* Hide default input label */
label[for="input_box"] {
    display: none;
}
input[data-baseweb="input"] {
    position: fixed;
    bottom: 1.5rem;
    left: 1rem;
    width: 75vw;
    z-index: 100;
}
button[kind="primary"] {
    position: fixed;
    bottom: 1.5rem;
    right: 1rem;
    z-index: 100;
}
</style>
""", unsafe_allow_html=True)

# === Input Field (Fixed Footer) ===
default_input = st.session_state.get("input_box", "")

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("", value=default_input, key="input_box", placeholder="Type your message here...")
    submitted = st.form_submit_button("Send")

if submitted and groq_api_key and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.markdown(f"**You:** {user_input}")

    try:
        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        # If file is uploaded, prepend its content to context
        if file_text.strip():
            context_message = {
                "role": "system",
                "content": f"The user uploaded a document. Use this information when helpful:\n{file_text[:3000]}"
            }
            st.session_state.chat_history.insert(1, context_message)

        # Call the model with streaming
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
        st.session_state.input_box = ""  # Clear stored input box

    except Exception as e:
        st.error(f"Error: {str(e)}")

elif not groq_api_key:
    st.warning("Please provide your Groq API Key in Streamlit secrets to start chatting.")
