import streamlit as st
import whisper
import tempfile
import torch
import os
import subprocess

# Check if GPU is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper model on the appropriate device (GPU or CPU)
model = whisper.load_model("small").to(device)

# Load Whisper model (cached to avoid reloading on each run)
@st.cache_resource
def load_model():
    return whisper.load_model("medium")

model = load_model()

# Function to extract audio from video
def extract_audio(video_path, output_audio_path):
    # FFmpeg command to extract audio as MP3
    command = [
        'ffmpeg', '-i', video_path, 
        '-vn', '-acodec', 'mp3', '-ar', '16000', '-ac', '1', 
        output_audio_path
    ]
    subprocess.run(command, check=True)

st.title("🎥 Video Speech Transcription with Whisper")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    # Save video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.getbuffer())
        video_path = temp_video.name

    # Extract audio from video
    audio_path = video_path.replace(".mp4", ".mp3")
    extract_audio(video_path, audio_path)

    st.write("Transcribing audio... ⏳")
    transcription = model.transcribe(audio_path)["text"]
    
    st.subheader("Transcription:")
    st.write(transcription)

    # Remove temporary files
    os.remove(video_path)
    os.remove(audio_path)
