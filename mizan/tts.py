import asyncio
import edge_tts
import streamlit as st
import concurrent.futures
import time
import os
import re
import tempfile

def clean_text_for_tts(text: str) -> str:
    """
   Clean and optimize the text to ensure smooth and natural speech synthesis.
    """
    if not text:
        return ""
    
    # 1. Remove emojis
    emoji_pattern = re.compile("["
                               u"\U00010000-\U0010ffff"  
                               u"\u2600-\u26ff"          
                               u"\u2700-\u27bf"          
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # 2. Remove markdown symbols
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'\*', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'^\s*[\-\*\+]\s+', '', text, flags=re.MULTILINE)
    text, _ = re.subn(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)
    text = text.replace('`', '').replace('_', ' ')
    
    # 3. Improve sentence flow to prevent awkward pauses
    text = text.replace('-\n', ' ')
    text = re.sub(r'\n+', '. ', text) 
    text = re.sub(r'\s+', ' ', text)   
    
    return text.strip()


async def generate_speech_async(text, output_file_path):
    cleaned_text = clean_text_for_tts(text)
    voice = "ar-SA-HamedNeural"  # Preferred voice
    communicate = edge_tts.Communicate(cleaned_text, voice)
    await communicate.save(output_file_path)

def text_to_speech(text, output_file_path=None):
    """
    Generate speech and securely save it in the system temp folder to avoid file accumulation in the project.
    """
    if not output_file_path:
        # Use system temp directory instead of the project directory
        temp_dir = tempfile.gettempdir()
        output_file_path = os.path.join(temp_dir, f"mizan_tts_{int(time.time() * 1000)}.mp3")
        
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(asyncio.run, generate_speech_async(text, output_file_path)).result()
        else:
            asyncio.run(generate_speech_async(text, output_file_path))
            
        return output_file_path
    except Exception as e:
        st.error(f"Error during TTS generation: {e}")
        return None