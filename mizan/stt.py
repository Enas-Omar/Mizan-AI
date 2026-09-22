import os
import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_stt_model():
    """
    Load the fine-tuned Libyan Whisper model from the local project directory.
    """
    # Define the local directory path for the Libyan dialect model
    local_model_path = "./models/whisper-small-libyan"
    
    # Check if the local directory exists, otherwise fallback to Tasneem's model online as a backup
    model_path = local_model_path if os.path.exists(local_model_path) else "Tass02/whisper-small-libyan"
    
    # Load the pipeline passing the local model
    stt_pipeline = pipeline("automatic-speech-recognition", model=model_path)
    return stt_pipeline

def transcribe_audio(audio_file_path):
    """
    Transcribe the recorded audio file into text using the Libyan dialect STT pipeline.
    """
    try:
        stt_model = load_stt_model()
        
        # Configure generation parameters for Arabic to achieve precise Libyan dialect transcription
        result = stt_model(
            audio_file_path, 
            generate_kwargs={"language": "arabic", "task": "transcribe"}
        )
        
        return result.get("text", "")
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""