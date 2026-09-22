import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Define the model ID for Tasneem's Libyan Arabic Whisper model and the local directory
model_id = "Tass02/whisper-small-libyan"
local_dir = "./models/whisper-small-libyan"

print(f"🔄 Loading and caching the Libyan Arabic model ({model_id}) locally...")

# Download the processor and model from Hugging Face
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)

# Save the files locally within the project directory
os.makedirs(local_dir, exist_ok=True)
processor.save_pretrained(local_dir)
model.save_pretrained(local_dir)

print("✅ Model downloaded, saved successfully within the project, and is now fully ready!")