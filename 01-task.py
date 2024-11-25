import streamlit as st
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

API_KEY = st.secrets("API_KEY")  
AZURE_ENDPOINT = st.secrets("AZURE_ENDPOINT")  

WHISPER_API = st.secrets("WHISPER_API")  
WHISPER_ENDPOINT = st.secrets("WHISPER_ENDPOINT")  

EMBED_API = st.secrets("EMBED_API")  
EMBED_ENDPOINT = st.secrets("EMBED_ENDPOINT")  

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
        "Ocp-Apim-Subscription-Key": API_KEY,
        "Content-Type": "audio/wav"  # Change this if the audio format is different
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
        "api-key": API_KEY
    }
    payload = {"input": text}
    try:
        url = f"{EMBED_ENDPOINT}"
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("data", [{}])[0].get("embedding", [])
    except Exception as e:
        return f"Error: {str(e)}\nURL: {url}"

# Streamlit App
# Streamlit App
st.title("Task 01 : Connectivity with GPT-4o")

# Buttons for task selection
col1, col2, col3 = st.columns(3)

with col1:
    chat_button = st.button("Chat Completion")
with col2:
    speech_button = st.button("Speech-to-Text")
with col3:
    embeddings_button = st.button("Text Embeddings")

# Chat Completion
if chat_button:
    st.header("Chat Completion")
    user_input = st.text_area("Enter your query:")
    if st.button("Get Chat Response"):
        if user_input.strip():
            response = query_chat_completion(user_input.strip())
            st.write("Response:", response)
        else:
            st.warning("Please enter some text.")

# Speech-to-Text
if speech_button:
    st.header("Speech-to-Text")
    st.write("Upload an audio file to transcribe:")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])
    if st.button("Transcribe Audio"):
        if uploaded_file:
            # Save the uploaded file temporarily
            temp_file_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
            
            # Query the Whisper API
            transcription_result = query_speech_to_text(temp_file_path)
            if transcription_result:
                st.write("Transcription Result:", transcription_result)
            else:
                st.error("Failed to transcribe the audio.")
            
            # Optionally delete the temp file
            os.remove(temp_file_path)
        else:
            st.warning("Please upload a valid audio file.")

# Text Embeddings
if embeddings_button:
    st.header("Text Embeddings")
    user_input = st.text_area("Enter text to generate embeddings:")
    if st.button("Generate Embeddings"):
        if user_input.strip():
            embeddings_result = query_text_embeddings(user_input.strip())
            if embeddings_result:
                st.write("Generated Embedding Vector:")
                st.json(embeddings_result)
            else:
                st.error("Failed to generate embeddings.")
        else:
            st.warning("Please enter some text.")