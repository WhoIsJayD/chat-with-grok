import json
import os
from datetime import datetime
from typing import Generator

import streamlit as st
from groq import Groq

# Page configuration
st.set_page_config(
    page_icon="🏎️",
    layout="wide",
    page_title="Groq Chat Advanced",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .chat-container {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e6f7ff;
    }
    .sidebar-content {
        padding: 10px;
    }
    .main-header {
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 style='text-align: center;'>🏎️ Groq Chat Streamlit App</h1>", unsafe_allow_html=True)
st.divider()

# Initialize Groq client
client = Groq(api_key=st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "")))

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama-3.1-8b-instant"
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = {}
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are Grok, a highly intelligent and context-aware assistant. "
        "Maintain conversation context by referencing previous messages when relevant. "
        "Provide clear, concise, and accurate responses tailored to the user's intent."
        "ALso You are an expert coding assistant for Bash, Perl, and Tcl. Always deliver clean, efficient, "
        "production-ready code with comments when needed. Prefer robust, idiomatic solutions. Make smart assumptions "
        "when requests are vague. No mistakes, no apologies—just solutions."
    )
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "token_count" not in st.session_state:
    st.session_state.token_count = {"prompt": 0, "completion": 0, "total": 0}

# Model details
models = {
    "meta-llama/llama-4-scout-17b-16e-instruct": {"name": "LLaMA 4 Scout 17B 16e Instruct", "tokens": 8192,
                                                  "developer": "Meta"},
    "meta-llama/llama-4-maverick-17b-128e-instruct": {"name": "LLaMA 4 Maverick 17B 128e Instruct", "tokens": 8192,
                                                      "developer": "Meta"},
    "qwen-qwq-32b": {"name": "Qwen QWQ 32B", "tokens": 8192, "developer": "Alibaba Cloud"},
    "mistral-saba-24b": {"name": "Mistral Saba 24B", "tokens": 32000, "developer": "Mistral"},
    "qwen-2.5-coder-32b": {"name": "Qwen 2.5 Coder 32B", "tokens": 8192, "developer": "Alibaba Cloud"},
    "qwen-2.5-32b": {"name": "Qwen 2.5 32B", "tokens": 8192, "developer": "Alibaba Cloud"},
    "deepseek-r1-distill-qwen-32b": {"name": "DeepSeek R1 Distill Qwen 32B", "tokens": 8192, "developer": "DeepSeek"},
    "deepseek-r1-distill-llama-70b": {"name": "DeepSeek R1 Distill LLaMA 70B", "tokens": 8192,
                                      "developer": "DeepSeek"},
    "llama-3.3-70b-specdec": {"name": "LLaMA 3.3 70B SpecDec", "tokens": 8192, "developer": "Meta"},
    "llama-3.2-1b-preview": {"name": "LLaMA 3.2 1B Preview", "tokens": 8192, "developer": "Meta"},
    "llama-3.2-3b-preview": {"name": "LLaMA 3.2 3B Preview", "tokens": 8192, "developer": "Meta"},
    "llama-3.2-11b-vision-preview": {"name": "LLaMA 3.2 11B Vision Preview", "tokens": 8192, "developer": "Meta"},
    "llama-3.2-90b-vision-preview": {"name": "LLaMA 3.2 90B Vision Preview", "tokens": 8192, "developer": "Meta"},
    "allam-2-7b": {"name": "Allam 2 7B", "tokens": 4096, "developer": "Saudi Data and AI Authority (SDAIA)"},
    "gemma2-9b-it": {"name": "Gemma2-9b-it", "tokens": 8192, "developer": "Google"},
    "llama-3.3-70b-versatile": {"name": "LLaMA 3.3 70B Versatile", "tokens": 8192, "developer": "Meta"},
    "llama-3.1-8b-instant": {"name": "LLaMA 3.1 8B Instant", "tokens": 8192, "developer": "Meta"},
    "llama-guard-3-8b": {"name": "LLaMA Guard 3 8B", "tokens": 8192, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"}
}

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")

    # System prompt
    st.subheader("System Prompt")
    system_prompt = st.text_area(
        "Customize the system prompt:",
        value=st.session_state.system_prompt,
        height=100
    )
    if system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt
        # Update system message in current conversation
        if st.session_state.messages and st.session_state.messages[0]["role"] == "system":
            st.session_state.messages[0]["content"] = system_prompt

    # Temperature
    st.subheader("Temperature")
    temperature = st.slider(
        "Adjust randomness:",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Higher values make output more random, lower values more deterministic."
    )
    st.session_state.temperature = temperature

    # Conversation management
    st.subheader("Conversation Management")
    if st.button("New Conversation"):
        st.session_state.messages = []
        st.session_state.current_conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
        st.session_state.token_count = {"prompt": 0, "completion": 0, "total": 0}
        st.rerun()

    if st.button("Save Conversation"):
        if st.session_state.messages:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.conversation_history[timestamp] = {
                "id": st.session_state.current_conversation_id,
                "messages": st.session_state.messages.copy(),
                "model": st.session_state.selected_model
            }
            st.success(f"Conversation saved at {timestamp}")

    if st.session_state.conversation_history:
        st.subheader("Saved Conversations")
        selected_conversation = st.selectbox(
            "Load a conversation:",
            options=list(st.session_state.conversation_history.keys()),
            format_func=lambda x: f"{x} ({st.session_state.conversation_history[x]['model']})"
        )
        if st.button("Load Selected"):
            selected_data = st.session_state.conversation_history[selected_conversation]
            st.session_state.messages = selected_data["messages"].copy()
            st.session_state.selected_model = selected_data["model"]
            st.session_state.current_conversation_id = selected_data["id"]
            st.rerun()

    if st.session_state.messages and st.button("Export as JSON"):
        conversation_data = {
            "id": st.session_state.current_conversation_id,
            "model": st.session_state.selected_model,
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(conversation_data, indent=2),
            file_name=f"conversation_{st.session_state.current_conversation_id}.json",
            mime="application/json"
        )

    # Token usage
    st.subheader("Token Usage")
    st.metric("Prompt Tokens", st.session_state.token_count["prompt"])
    st.metric("Completion Tokens", st.session_state.token_count["completion"])
    st.metric("Total Tokens", st.session_state.token_count["total"])

    # About
    st.subheader("About")
    st.markdown("Advanced chat app powered by Groq and Streamlit.")

# Main layout
col1, col2 = st.columns([1, 1])
with col1:
    model_option = st.selectbox(
        "Select Model:",
        options=list(models.keys()),
        format_func=lambda x: f"{models[x]['name']} ({models[x]['developer']})",
        index=list(models.keys()).index(st.session_state.selected_model)
    )

with col2:
    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512,
        max_value=models[model_option]["tokens"],
        value=min(32768, models[model_option]["tokens"]),
        step=512
    )

# Handle model change
if st.session_state.selected_model != model_option:
    if st.session_state.messages:
        st.warning("Changing models will clear the current conversation. Proceed?")
        if st.button("Confirm Model Change"):
            st.session_state.messages = []
            st.session_state.selected_model = model_option
            st.session_state.token_count = {"prompt": 0, "completion": 0, "total": 0}
            st.rerun()
    else:
        st.session_state.selected_model = model_option


# Token estimation
def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1 if text else 0


# Generate chat responses
def generate_chat_responses(chat_completion: Generator) -> Generator[str, None, None]:
    completion_tokens = 0
    for chunk in chat_completion:
        content = chunk.choices[0].delta.content or ""
        completion_tokens += estimate_tokens(content)
        yield content
    st.session_state.token_count["completion"] += completion_tokens
    st.session_state.token_count["total"] = st.session_state.token_count["prompt"] + st.session_state.token_count[
        "completion"]


# Display chat messages
st.subheader("Chat")
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👨‍💻"):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Type your message..."):
    # Initialize system message if needed
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "system", "content": st.session_state.system_prompt})

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👨‍💻"):
        st.markdown(prompt)

    # Estimate tokens
    prompt_tokens = estimate_tokens(prompt)
    st.session_state.token_count["prompt"] += prompt_tokens
    st.session_state.token_count["total"] += prompt_tokens

    # Generate response
    try:
        chat_messages = st.session_state.messages.copy()
        with st.spinner("Generating response..."):
            chat_completion = client.chat.completions.create(
                model=model_option,
                messages=chat_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )

        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in generate_chat_responses(chat_completion):
                full_response += chunk
                response_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.messages.pop()  # Remove failed user message
