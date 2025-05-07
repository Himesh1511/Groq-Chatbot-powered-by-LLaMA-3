import streamlit as st
from openai import AsyncOpenAI, OpenAI

# === Page Configuration ===
st.set_page_config(
    page_title="Groq LLaMA 3 Chatbot",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Sidebar for User Configuration ===
with st.sidebar:
    st.header("🔧 Configuration")
    groq_api_key = st.text_input("🔑 API Key", type="password", help="Enter your Groq API Key")
    st.markdown("---")
    st.write("🟢 **Model Details:**")
    st.text("LLaMA 3 - 8B / 70B-8192 Context")

# === Main Chat Interface ===
st.title("🦙 Groq Chatbot powered by LLaMA 3")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}
    ]

# Display past chat messages
st.write("### Chat")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        st.markdown(f"**Assistant:** {message['content']}")

# Input message box
user_input = st.text_input("Type your message here...", key="input_box")

if st.button("Send") and groq_api_key and user_input:
    # Display user input
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.markdown(f"**You:** {user_input}")

    try:
        # Initialize OpenAI client
        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        # Create chat completion
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Choose model variant
            messages=st.session_state.chat_history,
            stream=True
        )

        # Stream assistant response
        assistant_response = ""
        response_container = st.empty()
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            assistant_response += content
            response_container.markdown(f"**Assistant:** {assistant_response}")

        # Save assistant response to session state
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    except Exception as e:
        st.error(f"Error: {str(e)}")

elif not groq_api_key:
    st.warning("Please provide your API Key in the sidebar to start chatting.")
