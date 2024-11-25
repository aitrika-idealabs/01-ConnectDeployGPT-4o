import streamlit as st
import requests
import os

# Load secrets
API_KEY = st.secrets["API_KEY"]  
AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
WHISPER_API = st.secrets["WHISPER_API"]
WHISPER_ENDPOINT = st.secrets["WHISPER_ENDPOINT"]
EMBED_API = st.secrets["EMBED_API"]
EMBED_ENDPOINT = st.secrets["EMBED_ENDPOINT"]

# Azure OpenAI Chat Completion
def query_chat_completion(prompt):
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 800
    }
    try:
        url = f"{AZURE_ENDPOINT}"
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', 'No response')
    except Exception as e:
        return f"Error: {str(e)}\nURL: {url}"

# Azure Whisper Speech-to-Text
def query_speech_to_text(audio_file):
    headers = {
        "Ocp-Apim-Subscription-Key": WHISPER_API,
        "Content-Type": "audio/wav"
    }
    try:
        url = f"{WHISPER_ENDPOINT}"
        with open(audio_file, "rb") as file_data:
            response = requests.post(url, headers=headers, data=file_data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return f"Error: {str(e)}\nURL: {url}"

# Azure Text Embeddings
def query_text_embeddings(text):
    headers = {
        "Content-Type": "application/json",
        "api-key": EMBED_API
    }
    payload = {"input": text}
    try:
        url = f"{EMBED_ENDPOINT}"
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("data", [{}])[0].get("embedding", [])
    except Exception as e:
        return f"Error: {str(e)}\nURL: {url}"

# Initialize session state for buttons
if "task" not in st.session_state:
    st.session_state.task = None
if "chat_response" not in st.session_state:
    st.session_state.chat_response = ""
if "transcription_result" not in st.session_state:
    st.session_state.transcription_result = None
if "embedding_result" not in st.session_state:
    st.session_state.embedding_result = None

# Streamlit App
st.title("Azure AI Unified App")

# Task selection buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Chat Completion"):
        st.session_state.task = "chat"
with col2:
    if st.button("Speech-to-Text"):
        st.session_state.task = "speech"
with col3:
    if st.button("Text Embeddings"):
        st.session_state.task = "embeddings"

# Chat Completion Task
if st.session_state.task == "chat":
    st.header("Chat Completion")
    user_input = st.text_area("Enter your query:")
    if st.button("Get Chat Response", key="chat_response_button"):
        if user_input.strip():
            st.session_state.chat_response = query_chat_completion(user_input.strip())
        else:
            st.warning("Please enter some text.")
    if st.session_state.chat_response:
        st.write("Response:", st.session_state.chat_response)

# Speech-to-Text Task
elif st.session_state.task == "speech":
    st.header("Speech-to-Text")
    uploaded_file = st.file_uploader("Upload an audio file:", type=["wav", "mp3", "m4a"])
    if st.button("Transcribe Audio", key="transcribe_audio_button"):
        if uploaded_file:
            temp_file_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
            
            st.session_state.transcription_result = query_speech_to_text(temp_file_path)
            os.remove(temp_file_path)  # Clean up temp file
        else:
            st.warning("Please upload a valid audio file.")
    if st.session_state.transcription_result:
        st.write("Transcription Result:", st.session_state.transcription_result)

# Text Embeddings Task
elif st.session_state.task == "embeddings":
    st.header("Text Embeddings")
    user_input = st.text_area("Enter text to generate embeddings:")
    if st.button("Generate Embeddings", key="generate_embeddings_button"):
        if user_input.strip():
            st.session_state.embedding_result = query_text_embeddings(user_input.strip())
        else:
            st.warning("Please enter some text.")
    if st.session_state.embedding_result:
        st.write("Generated Embedding Vector:")
        st.json(st.session_state.embedding_result)
