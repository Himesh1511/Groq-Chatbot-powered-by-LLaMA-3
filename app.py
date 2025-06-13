import streamlit as st
from openai import OpenAI
import time
import PyPDF2
import io
from PIL import Image  # NEW: For image processing
import base64          # NEW: For image encoding

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

    # === Image Upload ===
    st.subheader("Upload Image")
    image_file = st.file_uploader("Upload an image for analysis", type=["png", "jpg", "jpeg"])
    if image_file:
        try:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode()
            # Save base64 and name in session for use in main chat
            st.session_state['uploaded_image_b64'] = img_b64
            st.session_state['uploaded_image_name'] = image_file.name
            st.success("Image uploaded and ready for analysis.")
        except Exception as e:
            st.error(f"Failed to process image: {e}")

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

# Simulate pre-filled input
user_input = st.chat_input(input_prompt, key="chat_input")
if user_input is None and default_input:
    st.session_state.pending_input = default_input
    st.rerun()

if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

# === Handle Input and Generate Assistant Response ===
if user_input and groq_api_key:
    # Append user's message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Prepare messages for the assistant
    messages = st.session_state.chat_history.copy()

    # If an image was uploaded, add it as a "tool" message for multimodal models
    image_b64 = st.session_state.get('uploaded_image_b64')
    image_name = st.session_state.get('uploaded_image_name')

    if image_b64 and image_name:
        # Add a placeholder system message (adjust as needed for your backend support)
        messages.append({
            "role": "user",
            "content": f"Analyze this image: {image_name}. If relevant, use the image to answer my question.",
            "image": image_b64  # Not a standard OpenAI field; for demonstration and backend adaptation.
        })

    # Generate assistant response in same cycle
    try:
        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        # If your backend supports vision/multimodal, adapt the API call accordingly.
        # For demonstration, send the image as part of the prompt/context.
        response = client.chat.completions.create(
            model="llama3-8b-8192",
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

        # Append assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Clear the uploaded image after analysis (optional)
        if 'uploaded_image_b64' in st.session_state:
            del st.session_state['uploaded_image_b64']
            del st.session_state['uploaded_image_name']

        st.rerun()

    except Exception as e:
        st.error(f"Error: {str(e)}")

elif not groq_api_key:
    st.warning("Please provide your Groq API Key in Streamlit secrets to start chatting.")
